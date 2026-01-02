import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np



device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "EleutherAI/gpt-neo-2.7B"




print(f"Loading model on {device}...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device)
print("Model loaded successfully!")



print(model)





