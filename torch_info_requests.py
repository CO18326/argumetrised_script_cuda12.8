import torch
from torchinfo import summary
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np



device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "EleutherAI/gpt-neo-2.7B"




#print(f"Loading model on {device}...")
tokenizer = AutoTokenizer.from_pretrained(model_path,torch_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device)
print("Model loaded successfully!")
batch_size = 8
seq_length = 1024
sample_input = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_length)).to(device)
# Generate detailed summary
model_summary = summary(
    model, 
    input_data=sample_input,
    col_names=["input_size", "output_size", "num_params", "trainable"],
    row_settings=["var_names"],
    dtypes=[torch.bfloat16],
    verbose=1,
    depth=3
)




print(model_summary)