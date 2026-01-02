import torch
import time
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    get_linear_schedule_with_warmup
)

from functools import partial
import ctypes
from torch.optim import AdamW
import weakref
from cpu_adm import DeepSpeedCPUAdam as ds_opt
import psutil
import threading
import os
import csv
import argparse
import re
import logging

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
#--------------------------------------------------Memory------------------------------------------------------
def human_readable_mb(x_bytes):
    mb = x_bytes / (1024 ** 2)
    return f"{mb:.2f}"

def predict_peak_memory(model, batch_size, seq_len, bf16=True, extra_safety=1.0):
    """
    Predict a conservative GPU memory peak (bytes) assuming bf16 (2 bytes per element).
    - model: HF model object (to read config and parameter count)
    - batch_size, seq_len: ints
    - extra_safety: multiplicative safety factor for activations/temps (>=1.0)
    Returns a dict of components in bytes.
    """
    # element size
    if bf16:
        elem_bytes = 2
    else:
        elem_bytes = 4  # fp32

    # total params (number)
    try:
        total_params = sum(p.numel() for p in model.parameters())
    except Exception:
        # fallback to config if parameters not yet loaded
        total_params = getattr(model.config, "n_parameters", None) or 0

    # param memory (model weights)
    param_bytes = int(total_params * elem_bytes)

    # gradients stored in same dtype as params in your setup (bf16), so approx same size
    grad_bytes = param_bytes

    # optimizer states: conservative default 2x parameter memory (e.g., Adam-like: m/v)
    optim_state_factor = 2.0
    optim_bytes = int(param_bytes * optim_state_factor)

    # try to get shape info from model config
    cfg = getattr(model, "config", None)
    hidden_size = None
    num_layers = None
    if cfg is not None:
        hidden_size = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None) or getattr(cfg, "d_model", None)
        num_layers = getattr(cfg, "n_layer", None) or getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer", None)

    # fallback guesses if not available
    if hidden_size is None:
        # try to infer from parameter shapes
        for n, p in model.named_parameters():
            if "embed" in n and p.ndim == 2:
                hidden_size = p.shape[1]
                break
    if hidden_size is None:
        hidden_size = 2048  # safe default

    if num_layers is None:
        # try to infer number of transformer blocks from modules
        try:
            num_layers = sum(1 for m in model.modules() if m.__class__.__name__.lower().startswith("transformer") or m.__class__.__name__.lower().startswith("block"))
            if num_layers == 0:
                num_layers = getattr(cfg, "n_layer", None) or 24
        except Exception:
            num_layers = 24

    # Activation estimate:
    # conservative formula: activations ~ batch * seq_len * hidden_size * num_layers * elem_bytes
    # plus extra for attention caches / intermediate buffers; multiply by activation_overhead_factor
    activation_base = int(batch_size * seq_len * hidden_size * num_layers * elem_bytes)

    # Add attention / softmax / intermediate buffers estimate: ~0.5x to 2.0x depending on model
    activation_overhead_factor = 1.5

    activation_bytes = int(activation_base * activation_overhead_factor * extra_safety)

    # working/other overhead (10% of total params+grads+optim+activations)
    other_overhead = int(0.10 * (param_bytes + grad_bytes + optim_bytes + activation_bytes))

    total_bytes = param_bytes + grad_bytes + optim_bytes + activation_bytes + other_overhead

    return {
        "total_params": total_params,
        "param_bytes": param_bytes,
        "grad_bytes": grad_bytes,
        "optim_bytes": optim_bytes,
        "activation_bytes": activation_bytes,
        "other_overhead": other_overhead,
        "total_bytes": total_bytes,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "elem_bytes": elem_bytes
    }

def print_memory_prediction(model, batch_size, seq_len, bf16=True, safety=1.5):
    d = predict_peak_memory(model, batch_size, seq_len, bf16=bf16, extra_safety=safety)
    print("\n=== MEMORY PREDICTION (bf16) ===")
    print(f"Model name: {getattr(model, 'name_or_path', getattr(model, '__class__', type(model)).__name__)}")
    print(f"Total params: {d['total_params']:,}  (approx)")
    print(f"Param memory (bf16): {human_readable_mb(d['param_bytes'])}")
    print(f"Gradient memory (bf16, approx): {human_readable_mb(d['grad_bytes'])}")
    print(f"Optimizer states (bf16, assumed 2x): {human_readable_mb(d['optim_bytes'])}")
    print(f"Activation estimate (batch={batch_size}, seq_len={seq_len}, hidden={d['hidden_size']}, layers={d['num_layers']}): {human_readable_mb(d['activation_bytes'])}")
    print(f"Working overhead (~10%): {human_readable_mb(d['other_overhead'])}")
    print("-" * 46)
    print(f"Estimated TOTAL memory (all on GPU, bf16): {human_readable_mb(d['total_bytes'])}")
    print(f"Estimated GPU peak (opt states on GPU): {human_readable_mb(d['total_bytes'])}")
    print("=== END PREDICTION ===\n")
#--------------------------------------------------Memory prediction end---------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="CUDA Prefetch Trainer Script")

    parser.add_argument("--model_name", type=str,
                        default="EleutherAI/gpt-neo-2.7B",
                        help="Model name or path to load")

    parser.add_argument("--seq_len", type=int,
                        default=512,
                        help="Sequence length")

    parser.add_argument("--steps", type=int,
                        default=100,
                        help="Max training steps")

    parser.add_argument("--batch_size", type=int,
                        default=1,
                        help="Per-device batch size")

    parser.add_argument("--prefetch_layers", type=int,
                        default=1,
                        help="PREFETCH_LAYERS_AHEAD value")

    parser.add_argument("--optimisation", type=int,
                        default=0,
                        help="0->nothing, 1->all activation pinned, 2->layered activation (specify number of layers in num_layer_pinned )")

    parser.add_argument("--prefetching", type=int,
                         default=0,
                         help="enable prefetching"
                         )

    parser.add_argument(
        "--num_layer_pinned", type=int,
                         default=0,
                         help="Number of layers whose activation u want to pinned")
    parser.add_argument(
        "--act_mem_pinned", type=int,
                         default=0,
                         help="amount of activation memory in (GB) whose activation u want to pinned")
    parser.add_argument(
        "--weight_pinned", type=int,
                         default=0,
                         help="1->weight pinned, 0->not pinned")

    parser.add_argument(
        "--optimiser_prefetch", type=int,
                         default=0,
                         help="1->optimiser prefetch, 0->optimiser not prefetch")
    parser.add_argument(
        "--optimiser_offload", type=int,
                         default=0,
                         help="1->optimiser offload")
    parser.add_argument(
        "--activation_prefetch", type=int,
                         default=0,
                         help="1->activation_prefetch")
    parser.add_argument(
        "--logging", type=int,
                         default=0,
                         help="1->enable logger")
    
    parser.add_argument(
        "--backward_prefetch", type=int,
        default=0,help="1->backward prefetch"
    )

    parser.add_argument(
        "--oversubscription_factor",type=float,
        default=0,help="oversubscription factor you want to enforce"
    )

    parser.add_argument(
        "--prefetch_weights_only",type=float,
        default=0,help="prefetch weights only"
    )
    
    parser.add_argument(
        "--build_csv",type=float,
        default=0,help="building the csv"
    )


    return parser.parse_args()


# ---------------- Prefetch Library ----------------
my_lib = ctypes.CDLL("./prefetch_async.so")
my_lib.prefetch_memory.argtypes = [ctypes.c_ulong, ctypes.c_ulong, ctypes.c_int, ctypes.c_void_p]
my_lib.prefetch_memory.restype = ctypes.c_int
my_lib.pin_memory_hint.argtypes = [ctypes.c_ulong, ctypes.c_ulong, ctypes.c_int]
my_lib.pin_memory_hint.restype = ctypes.c_int
#my_lib.prefetch_memory_batch.argtypes = [ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_int, ctypes.c_void_p]
#my_lib.prefetch_memory_batch.restype = ctypes.c_int
my_lib.cuda_malloc.argtypes=[ctypes.c_ulong]
my_lib.cuda_malloc.restype=ctypes.c_int
try:
    my_lib.print_first_byte.restype = ctypes.c_int
except Exception:
    pass

weight_logger=None 
input_logger=None
optimizer_logger=None
gradient_logger=None
# ---------------- Config ----------------
streams=None
PREFETCH_LAYERS_AHEAD=1
#---------------------------------------------------------------------------------------------
def create_loggers(log_dir="logs_allocation_flexible"):
    global weight_logger
    global input_logger
    global optimizer_logger
    global gradient_logger
    os.makedirs(log_dir, exist_ok=True)

    def make_logger(name, filename):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(os.path.join(log_dir, filename))
        file_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        return logger

    weight_logger = make_logger("weight_logger", "weight.txt")
    input_logger = make_logger("input_logger", "input.txt")
    optimizer_logger = make_logger("optimizer_logger", "optimizer.txt")
    gradient_logger = make_logger("gradient_logger", "gradient.txt")

    return weight_logger, input_logger, optimizer_logger, gradient_logger



#----------------------------------------------------------------------------------------



def extract_num(s):
    m = re.search(r'\.(\d+)\.', s)
    return int(m.group(1)) if m else None


#----------------------- Trainer ----------------------------------------

class CUDASyncTrainer(Trainer):
    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        # Backward pass
        self.accelerator.backward(loss)
        #print("hello")
        torch.cuda.synchronize()

        return loss.detach() / self.args.gradient_accumulation_steps






class CustomAdamW(AdamW):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                for k, v in state.items():
                    if k != "step" and isinstance(v, torch.Tensor):
                        data_ptr = v.data_ptr()
                        size_in_bytes = v.nelement() * v.element_size()
                        my_lib.prefetch_memory(data_ptr, size_in_bytes, 1, ctypes.c_void_p(streams[2]))
        return super().step(closure)


def _prefetch_tensor_bytes(data_ptr, size_in_bytes, stream_idx):
    if size_in_bytes <= 0:
        return
    my_lib.prefetch_memory(data_ptr, size_in_bytes, 1, ctypes.c_void_p(streams[stream_idx % len(streams)]))

def prefetch_tensor_if_large(tensor, stream_idx=1, threshold_bytes=2 * 1024 * 1024):
    if tensor is None or not torch.is_tensor(tensor):
        return
    size_in_bytes = tensor.nelement() * tensor.element_size()
    if size_in_bytes > threshold_bytes:
        _prefetch_tensor_bytes(tensor.data_ptr() + threshold_bytes, size_in_bytes - threshold_bytes, stream_idx)

_offload_ids = set()
_lock = threading.Lock()

def mark_for_offload(t):
    with _lock:
        _offload_ids.add(id(t))

def should_offload(t):
    with _lock:
        return id(t) in _offload_ids

def unmark(tid):
    with _lock:
        _offload_ids.discard(tid)


def selective_forward_hook(module, inputs, output):
    if isinstance(output, torch.Tensor):
        mark_for_offload(output)
    elif isinstance(output, (list, tuple)):
        for o in output:
            if isinstance(o, torch.Tensor):
                mark_for_offload(o)



def attach_hooks_by_type(model,num_layer):
    count = 0
    for name, module in model.named_modules():
        cls = module.__class__.__name__
        if extract_num(name) is not None and extract_num(name) < num_layer:
            module.register_forward_hook(selective_forward_hook)
            print(f"[HOOK] {cls:25s} --> {name}")
            count += 1
    print(f"\nTotal hooks attached: {count}\n")





# ---------------- Forward Prefetch Hook ----------------
model_modules = None

def hook(module, input, output=None, layer_idx=None, total_layers=None):
    for inp in input:
        if torch.is_tensor(inp):
            prefetch_tensor_if_large(inp, stream_idx=3)

    global model_modules
    if model_modules is None or total_layers is None or layer_idx is None:
        return
    if layer_idx == 0:
        end = min(PREFETCH_LAYERS_AHEAD, total_layers)
        for j in range(end):
            next_layer = model_modules[j]
            if hasattr(next_layer, "weight"):
                w = getattr(next_layer, "weight", None)
                if w is not None and torch.is_tensor(w):
                    stream_id = 2 + (j % (len(streams) - 2))
                    prefetch_tensor_if_large(w, stream_idx=stream_id)
    else:
        next_idx = layer_idx + PREFETCH_LAYERS_AHEAD
        if next_idx < total_layers:
            next_layer = model_modules[next_idx]
            if hasattr(next_layer, "weight"):
                w = getattr(next_layer, "weight", None)
                if w is not None and torch.is_tensor(w):
                    stream_id = 2 + (next_idx % (len(streams) - 2))
                    prefetch_tensor_if_large(w, stream_idx=stream_id)

def hook_only_act(module, input, output=None, layer_idx=None, total_layers=None):
    for inp in input:
        if torch.is_tensor(inp):
            prefetch_tensor_if_large(inp, stream_idx=3)




def hook_only_weights(module, input, output=None, layer_idx=None, total_layers=None):
    global model_modules
    if model_modules is None or total_layers is None or layer_idx is None:
        return
    if layer_idx == 0:
        end = min(PREFETCH_LAYERS_AHEAD, total_layers)
        for j in range(end):
            next_layer = model_modules[j]
            if hasattr(next_layer, "weight"):
                w = getattr(next_layer, "weight", None)
                if w is not None and torch.is_tensor(w):
                    stream_id = 2 + (j % (len(streams) - 2))
                    prefetch_tensor_if_large(w, stream_idx=stream_id)
    else:
        next_idx = layer_idx + PREFETCH_LAYERS_AHEAD
        if next_idx < total_layers:
            next_layer = model_modules[next_idx]
            if hasattr(next_layer, "weight"):
                w = getattr(next_layer, "weight", None)
                if w is not None and torch.is_tensor(w):
                    stream_id = 2 + (next_idx % (len(streams) - 2))
                    prefetch_tensor_if_large(w, stream_idx=stream_id)





def prefetch_params(module):
    for name, param in module.named_parameters(recurse=False):
        if param is not None and torch.is_tensor(param):
            prefetch_tensor_if_large(param, stream_idx=1)


def add_pre_backward_hook(module,name):
    def fw_hook(mod, inp, out):
        saved_refs = [weakref.ref(x) for x in inp if torch.is_tensor(x)]
        
        def _make_hook(mod_ref, saved_refs):
            def _hook(grad):
                prefetch_params(mod_ref)
                for ref in saved_refs:
                    act = ref()
                    if act is None:
                        continue
                    prefetch_tensor_if_large(act, stream_idx=1)
                return grad
            return _hook

        if torch.is_tensor(out) and out.requires_grad:
            #print(name)
            out.register_hook(_make_hook(mod, saved_refs))
        elif isinstance(out, tuple):
            for o in out:
                if torch.is_tensor(o) and o.requires_grad:
                    #print(name)
                    o.register_hook(_make_hook(mod, saved_refs))

    module.register_forward_hook(fw_hook)


_offload_lock = threading.Lock()
_offloaded_bytes = 0

def _size_in_bytes(tensor):
    return tensor.nelement() * tensor.element_size()


class StepTimeCallback(TrainerCallback):
    def __init__(self):
        self.start = None
        self.step_times = []
        self.peak_mems = []
        self.peak_mems_pinned=[]
    def on_step_begin(self, args, state, control, **kwargs):
        self.start = time.time()
    def on_step_end(self, args, state, control, **kwargs):
        global _offloaded_bytes
        torch.cuda.synchronize()
        duration = time.time() - self.start
        self.step_times.append(duration)
        max_mem = torch.cuda.max_memory_reserved() / (1024 ** 2)
        self.peak_mems.append(max_mem)
        torch._C._cuda_endUvmAllocate()
        max_mem_pinned = torch.cuda.max_memory_reserved() / (1024 ** 2)
        self.peak_mems_pinned.append(max_mem_pinned)
        torch._C._cuda_beginUvmAllocate()
        
        print(f"[Step {state.global_step}] {duration:.3f} sec | Max GPU Mem: {max_mem:.2f} MB | Max Pinned GPU Mem: {max_mem_pinned:.2f} MB" )
        torch.cuda.reset_peak_memory_stats()
        with _offload_lock:
            _offloaded_bytes = 0

def log_optimizer_state_addresses(optimizer):
    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            state = optimizer.state[param]
            #print(state.items())
            for key, value in state.items():
                if key=="step":
                    continue
                optimizer_logger.info(f"Optimizer State: {key}, Address: {hex(value.data_ptr())}, Size: {value.nelement() * value.element_size()}")

class OptStateLoggerCallback(TrainerCallback):
    def __init__(self):
        self.trainer_ref = None
        self.optimizer_ref =None
    
    def on_optimizer_step(self, args, state, control, **kwargs):
        global model_modules
        if state.global_step==1:
            for i,module  in enumerate(model_modules):
                if hasattr(module, "weight"):
                    p=module.weight
                    if p.grad is not None:
                        gradient_logger.info(f"Module: {p.__class__.__name__}, InputGradient {1} Address: {hex(p.grad.data_ptr())}, Size: {p.grad.nelement() * p.grad.element_size()}")
    
    def on_step_end(self, args, state, control, **kwargs):
        global model_modules
        trainer = self.trainer_ref
        opt= trainer.optimizer
        if state.global_step==1:
            log_optimizer_state_addresses(opt)
            


def register_multi_layer_hooks(model,prefetch_weights=False, N=1,prefetch_weights_only=False):
    global model_modules
    #model.register_forward_pre_hook(partial(hook, layer_idx=i, total_layers=total)) if prefetch_weights else model.register_forward_pre_hook(partial(hook_only_act, layer_idx=i, total_layers=total))
    model_modules = list(model.modules())
    model_modules=[m for m in model_modules if hasattr(m, "weight") ]
    total = len(model_modules)
    for i, module in enumerate(model_modules):
        if prefetch_weights:
            module.register_forward_pre_hook(partial(hook, layer_idx=i, total_layers=total))  
        
        elif prefetch_weights_only:
            module.register_forward_pre_hook(partial(hook_only_weights, layer_idx=i, total_layers=total))

        else:
            module.register_forward_pre_hook(partial(hook_only_act, layer_idx=i, total_layers=total))
def register_backward_hook(model):
    global model_modules
    model_modules = list(model.modules())
    total = len(model_modules)
    #add_pre_backward_hook(model)
    for name, module in model.named_modules():
        add_pre_backward_hook(module,name)

def log_model_weight(model):
    global weight_logger
    global model_modules
    model_modules = list(model.modules())
    for i, module in enumerate(model_modules):
        if hasattr(module, "weight"):
            data_ptr = module.weight.data_ptr()
            size_in_bytes = module.weight.nelement() * module.weight.element_size()
            weight_logger.info(f"Address: {hex(data_ptr)}, Size: {size_in_bytes}")

        

def build_log_dir_name(args):
    parts = []
    for k, v in vars(args).items():
        if k == "oversubscription_factor":
            continue
        if v is None:
            continue
        if isinstance(v, (list, tuple, dict)):
            v = str(v).replace(" ", "_").replace("/", "-")
        else:
            v = str(v).replace(" ", "_").replace("/", "-")

        parts.append(f"{v}")
        
    filename = "-".join(parts)
    return filename


def main():
    args = parse_args()
    global streams
    global PREFETCH_LAYERS_AHEAD
    PREFETCH_LAYERS_AHEAD = args.prefetch_layers
    streams = [torch.cuda.Stream().cuda_stream for _ in range(PREFETCH_LAYERS_AHEAD + 3)]
    global _offloaded_bytes

    

    #model_name = "ibm-granite/granite-3.0-8b-base"
    model_name = args.model_name
    seq_len = args.seq_len
    steps = args.steps
    batch_size = args.batch_size
    optimisation= args.optimisation
    prefetching = args.prefetching
    activation_prefetch = args.activation_prefetch
    logging=args.logging
    backward_prefetch=args.backward_prefetch


    tokenizer = AutoTokenizer.from_pretrained(model_name,token="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")
    tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset("wikitext", "wikitext-2-v1", split="train")

    if logging:
        dir_name="page_faults_"+build_log_dir_name(args)
        create_loggers(dir_name)
    
    def tokenize_fn(examples):
        model_inputs = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=seq_len,
        )
        labels = []
        for seq in model_inputs["input_ids"]:
            labels.append([tok if tok != tokenizer.pad_token_id else -100 for tok in seq])
        model_inputs["labels"] = labels
        return model_inputs

    dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    if args.weight_pinned:
        torch._C._cuda_endUvmAllocate()
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16,token="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"
        ).cuda(0)
        torch._C._cuda_beginUvmAllocate()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16,token="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"
        ).cuda(0)
    
    #print_memory_prediction(model, batch_size, seq_len, bf16=True, safety=1.5)
    if logging:
        log_model_weight(model)
    if prefetching:
        register_multi_layer_hooks(model,True,PREFETCH_LAYERS_AHEAD)
    if args.prefetch_weights_only:
        register_multi_layer_hooks(model,False,PREFETCH_LAYERS_AHEAD,True)
    if backward_prefetch:
        register_backward_hook(model)
    if activation_prefetch:
        register_multi_layer_hooks(model,False)
    if optimisation==2:
        attach_hooks_by_type(model,args.num_layer_pinned)

    callbacks=[StepTimeCallback(),OptStateLoggerCallback() ] if logging else [StepTimeCallback()]
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=batch_size,
        num_train_epochs=1,
        max_steps=2,
        bf16=True,
        logging_dir="./logs",
        logging_steps=1,
        save_strategy="no",
        report_to="none",
    )

    
    trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=[callbacks[0]],
        
        )
    
    trainer.train()


    cb = callbacks[0]

    avg_step = sum(cb.step_times) / len(cb.step_times)
    peak_mem = max(cb.peak_mems)   # MB
    peak_mems_pinned = max(cb.peak_mems_pinned)

    free_memory, total_memory = torch.cuda.memory.mem_get_info()

    achieved_oversubscription_factor=((peak_mem+peak_mems_pinned)*1024*1024/total_memory)

    print(f"ACHIEVED OVERSUBSCRIPTION FACTOR {achieved_oversubscription_factor:.2f}")
        
    final_oversubscription=achieved_oversubscription_factor
    extra_memory_needed=0
    if args.oversubscription_factor and round(achieved_oversubscription_factor,2) > args.oversubscription_factor :
        print("Given oversubscription factor can not be achieved") 
    elif args.oversubscription_factor and round(achieved_oversubscription_factor,2) < args.oversubscription_factor:

        needed_memory=((peak_mem+peak_mems_pinned)*(1024**2)/args.oversubscription_factor)
        extra_memory_needed= int((total_memory - needed_memory)/(1024**3))
        print("EXTRA MEMORY NEEDED",extra_memory_needed)
        my_lib.cuda_malloc(extra_memory_needed)
        final_oversubscription=args.oversubscription_factor

    final_gpu_mem= (total_memory/(1024**2) if not extra_memory_needed else  needed_memory/(1024**2))


        
        # Model memory breakdown using existing function
    pred = predict_peak_memory(model, batch_size, seq_len, bf16=True, extra_safety=1.0)

    param_b = pred["param_bytes"]
    grad_b  = pred["grad_bytes"]
    optim_b = pred["optim_bytes"]

    activation_b = ((peak_mem+peak_mems_pinned) * 1024**2) - (param_b + grad_b + optim_b)
    activation_b = max(activation_b, 0)

    print("\n================ WARMUP SUMMARY ================")
    print(f"Average Step Time: {avg_step:.4f} sec")
    print(f"Peak GPU Memory  : {peak_mem:.2f} MB")
    print(f"Weights Memory   : {human_readable_mb(param_b)}")
    print(f"Gradients Memory : {human_readable_mb(grad_b)}")
    print(f"Optimizer Memory : {human_readable_mb(optim_b)}")
    print(f"Activation Memory: {human_readable_mb(activation_b)}")
    print("================================================\n")
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    callbacks[0].step_times=[]
    callbacks[0].peak_mems=[]
    callbacks[0].peak_mems_pinned=[]
    
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=batch_size,
        num_train_epochs=1,
        max_steps=steps,
        bf16=True,
        logging_dir="./logs",
        logging_steps=1,
        save_strategy="no",
        report_to="none",
    )

    optimizer = ds_opt(model.parameters(), lr=1e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_training_steps=20, num_warmup_steps=2
    )

    
    
    
    
    if args.optimiser_prefetch:
        
        optimizer=CustomAdamW(model.parameters(), lr=1e-5)
        
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=callbacks,
            optimizers=(optimizer, None),
        )
        

    elif args.optimiser_offload:
        
        trainer = CUDASyncTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=callbacks,
            optimizers=(optimizer, None),
        )
    
    else:
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=callbacks,
        
        )

    if len(callbacks)>1:
            callbacks[1].trainer_ref=trainer

    def pack_hook(tensor):
        global _offloaded_bytes
        if not torch.is_tensor(tensor):
            return tensor
        size = _size_in_bytes(tensor)
        if not tensor.is_cuda:
            return tensor
        if logging:
            input_logger.info(f"Module: INPUT LOG, Input {0} Address: {hex(tensor.data_ptr())}, Size: {tensor.nelement() * tensor.element_size()}")

        with _offload_lock:
            remaining = args.act_mem_pinned*1024 * (1024 ** 2) - _offloaded_bytes
            if size <= remaining:
                torch._C._cuda_endUvmAllocate()
                packed = tensor.detach().clone()
                torch._C._cuda_beginUvmAllocate()
                _offloaded_bytes += size
                return packed
            else:
                return tensor

    def pack_hook_layer(tensor):
        if logging:
            input_logger.info(f"Module: INPUT LOG, Input {0} Address: {hex(tensor.data_ptr())}, Size: {tensor.nelement() * tensor.element_size()}")
        if not should_offload(tensor):
            return tensor
        #print("inside the pack hook:",id(tensor))
        torch._C._cuda_endUvmAllocate()

        packed=tensor.detach().clone()
        torch._C._cuda_beginUvmAllocate()

        return packed
    
    def pack_hook_logging(tensor):
        if logging:
            input_logger.info(f"Module: INPUT LOG, Input {0} Address: {hex(tensor.data_ptr())}, Size: {tensor.nelement() * tensor.element_size()}")
        return tensor


    def unpack_hook(packed):
        return packed

    def build_csv_name(args):
        parts = []
        for k, v in vars(args).items():
            if k == "oversubscription_factor":
                continue
            if v is None:
                continue
            if isinstance(v, (list, tuple, dict)):
                v = str(v).replace(" ", "_").replace("/", "-")
            else:
                v = str(v).replace(" ", "_").replace("/", "-")

            parts.append(f"{v}")
        
        filename = "-".join(parts) + ".csv"
        return filename
    if optimisation==1 :
        print("check----12")
        with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            trainer.train()
    elif optimisation==2:
        with torch.autograd.graph.saved_tensors_hooks(pack_hook_layer, unpack_hook):
            trainer.train()
    else:
        with torch.autograd.graph.saved_tensors_hooks(pack_hook_logging, unpack_hook):
            
            trainer.train()
            

    #print("Training end")
    cb = callbacks[0]

    avg_step = sum(cb.step_times) / len(cb.step_times)
    peak_mem = max(cb.peak_mems)
    peak_mems_pinned = max(cb.peak_mems_pinned)   # MB

        # Model memory breakdown using existing function
    pred = predict_peak_memory(model, batch_size, seq_len, bf16=True, extra_safety=1.0)

    #final_oversubscription=0
    #final_gpu_mem=0



    param_b = pred["param_bytes"]
    grad_b  = pred["grad_bytes"]
    optim_b = pred["optim_bytes"]

    activation_b = ((peak_mem+peak_mems_pinned) * 1024**2) - (param_b + grad_b + optim_b)
    activation_b = max(activation_b, 0)

    print("\n================ FINAL SUMMARY ================")
    print(f"Average Step Time: {avg_step:.4f} sec")
    print(f"Peak GPU Memory  : {peak_mem:.2f} MB")
    print(f"Pinned Peak GPU Memory  : {peak_mems_pinned:.2f} MB")
    print(f"Weights Memory   : {human_readable_mb(param_b)}")
    print(f"Gradients Memory : {human_readable_mb(grad_b)}")
    print(f"Optimizer Memory : {human_readable_mb(optim_b)}")
    print(f"Activation Memory: {human_readable_mb(activation_b)}")
    print(f"Oversubscription Factor: {final_oversubscription:.2f}")
    print("================================================\n")

    if args.build_csv :
    
        csv_name = f"/modelops/priyanka/runtime_csvs/{build_csv_name(args)}"

        header = "avg_step_time,peak_gpu_mem,peak_gpu_mem_pinned,weights_mem,grads_mem,opt_mem,activation_mem,oversub_factor,available_gpu_mem\n"
        row = f"{avg_step:.4f},{peak_mem:.2f},{peak_mems_pinned:.2f},{human_readable_mb(param_b)},{human_readable_mb(grad_b)},{human_readable_mb(optim_b)},{human_readable_mb(activation_b)},{final_oversubscription:.2f},{final_gpu_mem:.2f}\n"

        write_header = not os.path.exists(csv_name)

        with open(csv_name, "a") as f:
            if write_header:
                f.write(header)
            f.write(row)

        print(f"\nSaved/updated CSV: {csv_name}\n")


if __name__ == "__main__":
    
    torch._C._cuda_beginUvmAllocate()
    torch.cuda.set_device('cuda:0')
    main()
    #my_lib.print_first_byte()
    torch._C._cuda_endUvmAllocate()
