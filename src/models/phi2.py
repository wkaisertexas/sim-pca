import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_default_device("cuda")

model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

inputs = tokenizer('''What is 1+2?''', return_tensors="pt", return_attention_mask=False)


with torch.cuda.nvtx.range("generation"):
   outputs = model.generate(**inputs, max_length=200)

text = tokenizer.batch_decode(outputs)[0]
print(text)
