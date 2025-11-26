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


def parse_args():
    parser = argparse.ArgumentParser(description="CUDA Prefetch Trainer Script")

    parser.add_argument("--model_name", type=str,
                        default="EleutherAI/gpt-neo-1.3B",
                        help="Model name or path to load")

    parser.add_argument("--seq_len", type=int,
                        default=512,
                        help="Sequence length")

    parser.add_argument("--steps", type=int,
                        default=100,
                        help="Max training steps")

    parser.add_argument("--batch_size", type=int,
                        default=2,
                        help="Per-device batch size")

    parser.add_argument("--prefetch_layers", type=int,
                        default=5,
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


    return parser.parse_args()


# ---------------- Prefetch Library ----------------
my_lib = ctypes.CDLL("./prefetch_async.so")
my_lib.prefetch_memory.argtypes = [ctypes.c_ulong, ctypes.c_ulong, ctypes.c_int, ctypes.c_void_p]
my_lib.prefetch_memory.restype = ctypes.c_int
my_lib.pin_memory_hint.argtypes = [ctypes.c_ulong, ctypes.c_ulong, ctypes.c_int]
my_lib.pin_memory_hint.restype = ctypes.c_int
my_lib.prefetch_memory_batch.argtypes = [ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_int, ctypes.c_void_p]
my_lib.prefetch_memory_batch.restype = ctypes.c_int
try:
    my_lib.print_first_byte.restype = ctypes.c_int
except Exception:
    pass

# ---------------- Config ----------------
PREFETCH_LAYERS_AHEAD = 4
streams = [torch.cuda.Stream().cuda_stream for _ in range(PREFETCH_LAYERS_AHEAD + 3)]

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


def prefetch_params(module):
    for name, param in module.named_parameters(recurse=False):
        if param is not None and torch.is_tensor(param):
            prefetch_tensor_if_large(param, stream_idx=1)


def add_pre_backward_hook(module):
    def fw_hook(mod, inp, out):
        saved_refs = [weakref.ref(x) for x in inp if torch.is_tensor(x)]

        def _make_hook(mod_ref, saved_refs):
            def _hook(grad):
                for ref in saved_refs:
                    act = ref()
                    if act is None:
                        continue
                    prefetch_tensor_if_large(act, stream_idx=1)
                return grad
            return _hook

        if torch.is_tensor(out) and out.requires_grad:
            print("check....")
            out.register_hook(_make_hook(mod, saved_refs))
        elif isinstance(out, tuple):
            for o in out:
                if torch.is_tensor(o) and o.requires_grad:
                    print("check....")
                    o.register_hook(_make_hook(mod, saved_refs))

    module.register_forward_hook(fw_hook)


_offload_lock = threading.Lock()
_offloaded_bytes = 0

def _size_in_bytes(tensor):
    return tensor.nelement() * tensor.element_size()


class StepTimeCallback(TrainerCallback):
    def __init__(self):
        self.start = None
    def on_step_begin(self, args, state, control, **kwargs):
        self.start = time.time()
    def on_step_end(self, args, state, control, **kwargs):
        global _offloaded_bytes
        torch.cuda.synchronize()
        duration = time.time() - self.start
        max_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"[Step {state.global_step}] {duration:.3f} sec | Max GPU Mem: {max_mem:.2f} MB")
        torch.cuda.reset_peak_memory_stats()
        with _offload_lock:
            _offloaded_bytes = 0

def register_multi_layer_hooks(model,prefetch_weights=False, N=PREFETCH_LAYERS_AHEAD):
    global model_modules
    model_modules = list(model.modules())
    total = len(model_modules)
    for i, module in enumerate(model_modules):
        module.register_forward_pre_hook(partial(hook, layer_idx=i, total_layers=total)) if prefetch_weights else module.register_forward_pre_hook(partial(hook_only_act, layer_idx=i, total_layers=total))


def main():
    args = parse_args()

    global PREFETCH_LAYERS_AHEAD
    PREFETCH_LAYERS_AHEAD = args.prefetch_layers

    global _offloaded_bytes

    #model_name = "ibm-granite/granite-3.0-8b-base"
    model_name = args.model_name
    seq_len = args.seq_len
    steps = args.steps
    batch_size = args.batch_size
    optimisation= args.optimisation
    prefetching = args.prefetching
    activation_prefetch = args.activation_prefetch


    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset("wikitext", "wikitext-2-v1", split="train")

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
            model_name, torch_dtype=torch.bfloat16,
        ).cuda()
        torch._C._cuda_beginUvmAllocate()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16,
        ).cuda()

    if prefetching:
        register_multi_layer_hooks(model,True)
    if activation_prefetch:
        register_multi_layer_hooks(model,False)
    if optimisation==2:
        attach_hooks_by_type(model,args.num_layer_pinned)

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
            callbacks=[StepTimeCallback()],
            optimizers=(optimizer, scheduler),
        )

    elif args.optimiser_offload:
        
        trainer = CUDASyncTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=[StepTimeCallback()],
            optimizers=(optimizer, None),
        )
    
    else:
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=[StepTimeCallback()],
        )

    def pack_hook(tensor):
        global _offloaded_bytes
        if not torch.is_tensor(tensor):
            return tensor
        size = _size_in_bytes(tensor)
        if not tensor.is_cuda:
            return tensor

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
        if not should_offload(tensor):
            return tensor
        #print("inside the pack hook:",id(tensor))
        torch._C._cuda_endUvmAllocate()

        packed=tensor.detach().clone()
        torch._C._cuda_beginUvmAllocate()

        return packed


    def unpack_hook(packed):
        return packed


    if optimisation==1 :
        print("check----12")
        with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            trainer.train()
    elif optimisation==2:
        with torch.autograd.graph.saved_tensors_hooks(pack_hook_layer, unpack_hook):
            trainer.train()
    else:
        trainer.train()


if __name__ == "__main__":
    torch._C._cuda_beginUvmAllocate()
    main()
    torch._C._cuda_endUvmAllocate()
