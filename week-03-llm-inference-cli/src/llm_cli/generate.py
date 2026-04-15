# Imports
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Generate text
#def generate_text(model_name)


# Device
device = 'cuda' if torch.cuda.is_available else 'cpu'
# Model name
model_name = "Qwen/Qwen2.5-7B-Instruct"
# Load model and Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype = torch.bfloat16, device_map= 'auto').to(device)

# Initialise conversation
message = [
    { "role":"system", "content":"You are a helpful assistant?"},
    {"role":"user", "content":"What is LLM Engineering about?"}
]

inputs = tokenizer.apply_chat_template(message,
tokenize =True, return_tensors="pt").to(device)

outputs = model.generate(**inputs,
max_new_tokens =100,temperature=0.7, do_sample=True)

print(outputs)
