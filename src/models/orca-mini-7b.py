from transformers import AutoModel, AutoTokenizer
import torch

model_slug = "pankajmathur/orca_mini_v7_72b"
model = AutoModel.from_pretrained(model_slug)
tokenizer = AutoTokenizer.from_pretrained(model_slug)
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "1 + 2 = "}
]
gen_input = tokenizer.apply_chat_template(messages, return_tensors="pt")

with torch.cuda.nvtx.range("generation"):
    model.generate(**gen_input)
