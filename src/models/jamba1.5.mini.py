"""
Code used to run 

https://huggingface.co/ai21labs/AI21-Jamba-1.5-Mini
"""

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import torch

model = "ai21labs/AI21-Jamba-1.5-Mini"
number_gpus = 1

llm = LLM(model=model,
          max_model_len=200*1024,
          tensor_parallel_size=number_gpus)

tokenizer = AutoTokenizer.from_pretrained(model)

messages = [
   {"role": "system", "content": "Your are a AI assistant"},
   {"role": "user", "content": "1 + 2 ="},
]

prompts = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

sampling_params = SamplingParams(temperature=0.4, top_p=0.95, max_tokens=100) 

with torch.cuda.nvtx.range("generation"):
   outputs = llm.generate(prompts, sampling_params)

generated_text = outputs[0].outputs[0].text
print(generated_text)