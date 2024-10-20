# Load model directly
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-70B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-70B-Instruct")


messages = [ 
    {"role": "system", "content": "You are a AI assistant."}, 
    {"role": "user", "content": "What is 1+2 = ?"}, 
] 

pipe = pipeline(    
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
) 

generation_args = { 
    "max_new_tokens": 500, 
    "return_full_text": False, 
    "temperature": 0.0, 
    "do_sample": False, 
} 


with torch.cuda.nvtx.range("generation"):
    output = pipe(messages, **generation_args) 

print(output[0]['generated_text'])
