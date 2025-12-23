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
# keep ds_opt import available (not used by default)
#from cpu_adm import DeepSpeedCPUAdam as ds_opt
import psutil
import threading
import os
import csv

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
PREFETCH_LAYERS_AHEAD = 2
streams = [torch.cuda.Stream().cuda_stream for _ in range(PREFETCH_LAYERS_AHEAD + 3)]

#---------------------------------------------------
class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, num_items_in_batch):
        # --- BEFORE LOSS / BACKWARD HOOK ---
        print(">>> BEFORE LOSS")

        loss = super().compute_loss(model, inputs, num_items_in_batch)

        # Inject custom code BEFORE backward()
        torch._C._cuda_endUvmAllocate()

        return (loss,)

# ---------------- Custom Optimizer ----------------
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

# ---------------- Utilities ----------------
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

# ---------------- Forward Prefetch Hook ----------------
model_modules = None

def hook(module, input, output=None, layer_idx=None, total_layers=None):



    '''
    input_ptrs=[inp.data_ptr() for inp in input if torch.is_tensor(inp)]
    size_arr=[inp.nelement() * inp.element_size() for inp in input if torch.is_tensor(inp)]
    if len(input_ptrs):
        C_LONG_ARRAY = ctypes.c_ulong * len(input_ptrs)

        c_array_address=C_LONG_ARRAY(*input_ptrs)

        c_size_arr=C_LONG_ARRAY(*size_arr)
        #print("hello_123")
        my_lib.prefetch_memory_batch(ctypes.addressof(c_array_address),ctypes.addressof(c_size_arr),len(input_ptrs),1, ctypes.c_void_p(streams[3 % len(streams)]))'''


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

# ---------------- Prefetch module parameters ----------------
def prefetch_params(module):
    for name, param in module.named_parameters(recurse=False):
        if param is not None and torch.is_tensor(param):
            prefetch_tensor_if_large(param, stream_idx=1)

# ---------------- Add Pre-Backward Hook ----------------
def add_pre_backward_hook(module):
    def fw_hook(mod, inp, out):
        saved_refs = [weakref.ref(x) for x in inp if torch.is_tensor(x)]

        def _make_hook(mod_ref, saved_refs):
            def _hook(grad):
                #prefetch_params(mod_ref)
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

# ---------------- Offload Budget Tracking ----------------
OFFLOAD_LIMIT_BYTES = 30*1024 * (1024 ** 2)  # 10 GB
_offload_lock = threading.Lock()
_offloaded_bytes = 0

def _size_in_bytes(tensor):
    return tensor.nelement() * tensor.element_size()

# ---------------- Step Timer ----------------
class StepTimeCallback(TrainerCallback):
    def __init__(self):
        self.start = None
    def on_step_begin(self, args, state, control, **kwargs):
        self.start = time.time()
    def on_step_end(self, args, state, control, **kwargs):
        global _offloaded_bytes
        torch.cuda.synchronize()
        duration = time.time() - self.start
        print(f"[Step {state.global_step}] {duration:.3f} sec")
        # ✅ Reset offload counter per step
        with _offload_lock:
            _offloaded_bytes = 0
    '''def on_pre_optimizer_step(self, args, state, control, **kwargs):
        torch._C._cuda_endUvmAllocate()
    def on_optimizer_step(self, args, state, control, **kwargs):
        torch._C._cuda_beginUvmAllocate()'''

# ---------------- Register Hooks ----------------
def register_multi_layer_hooks(model, N=PREFETCH_LAYERS_AHEAD):
    global model_modules
    model_modules = list(model.modules())
    total = len(model_modules)
    for i, module in enumerate(model_modules):
        module.register_forward_pre_hook(partial(hook, layer_idx=i, total_layers=total))
        #add_pre_backward_hook(module)

# ---------------- Main ----------------
def main():
    global _offloaded_bytes

    model_name = "ibm-granite/granite-3.0-8b-base"
    #model_name = "EleutherAI/gpt-neo-2.7B"
    #model_name = "facebook/opt-13b"
    seq_len = 1024
    steps = 100
    batch_size =16

    tokenizer = AutoTokenizer.from_pretrained(model_name,token="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")
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
    #torch._C._cuda_endUvmAllocate()
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16,token="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").cuda()
    #torch._C._cuda_beginUvmAllocate()
    register_multi_layer_hooks(model, PREFETCH_LAYERS_AHEAD)

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

    '''trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[StepTimeCallback()],

    )'''

    optimizer = CustomAdamW(model.parameters(), lr=1e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_training_steps=20, num_warmup_steps=2
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[StepTimeCallback()],
        optimizers=(optimizer, scheduler),
    )

    # ---------------- Saved Tensor Hook (Activation Offload up to 10GB) ----------------
    def pack_hook(tensor):
        '''global _offloaded_bytes
        if not torch.is_tensor(tensor):
            return tensor
        size = _size_in_bytes(tensor)
        if not tensor.is_cuda:
            return ("OFFLOADED", tensor, 0)

        with _offload_lock:
            remaining = OFFLOAD_LIMIT_BYTES - _offloaded_bytes
            if size <= remaining:
                #cpu_t = tensor.detach().to("cpu", non_blocking=True)
                data_ptr = tensor.data_ptr()
                my_lib.prefetch_memory(data_ptr, size, 2, ctypes.c_void_p(streams[4]))
                _offloaded_bytes += size
                return tensor
            else:
                return tensor'''
        global _offloaded_bytes
        if not torch.is_tensor(tensor):
            return tensor
        size = _size_in_bytes(tensor)
        if not tensor.is_cuda:
            return tensor

        with _offload_lock:
            remaining = OFFLOAD_LIMIT_BYTES - _offloaded_bytes
            if size <= remaining:
                #cpu_t = tensor.detach().to("cpu", non_blocking=True)
                torch._C._cuda_endUvmAllocate()
                #packed = torch.empty(
                #tensor.size(),
                #dtype=tensor.dtype,
                #layout=tensor.layout,
                #device="cuda"
                #)
                #packed.copy_(tensor.detach())
                packed=tensor.detach().clone()
                torch._C._cuda_beginUvmAllocate()
                _offloaded_bytes += size
                return packed
            else:
                return tensor


        #return packed

    def unpack_hook(packed):
        #size = _size_in_bytes(packed)
        #data_ptr = packed.data_ptr()
        #my_lib.prefetch_memory(data_ptr, size, 1, ctypes.c_void_p(streams[6]))
        return packed


    #with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    trainer.train()

# ---------------- Run ----------------
if __name__ == "__main__":
    torch._C._cuda_beginUvmAllocate()
    main()
    torch._C._cuda_endUvmAllocate()
