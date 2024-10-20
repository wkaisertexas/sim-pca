"""
code used to run:

https://huggingface.co/microsoft/Phi-3.5-MoE-instruct
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline 

torch.random.manual_seed(0) 

model = AutoModelForCausalLM.from_pretrained( 
    "microsoft/Phi-3.5-MoE-instruct",  
    device_map="cuda",  
    torch_dtype="auto",  
    trust_remote_code=True,  
) 

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-MoE-instruct") 

messages = [ 
    {"role": "system", "content": "You are a AI assistant"}, 
    {"role": "user", "content": "What is 1+2?"}
] 

pipe = pipeline( 
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
) 

generation_args = { 
    "max_new_tokens": 100, 
    "return_full_text": False, 
    "temperature": 0.0, 
    "do_sample": False, 
} 

with torch.cuda.nvtx.range("generation"):
    output = pipe(messages, **generation_args) 

print(output[0]['generated_text'])
