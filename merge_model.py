import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# PATHS
base_model_id = "huihui-ai/Qwen2.5-Coder-3B-Instruct-abliterated"
adapter_path = "./qwen-uncensored-l4/checkpoint-200" 
output_dir = "./qwen-merged-final"
"""
# Update above as follows:
#
# base_model_id : can be the id after .com of (https://huggingface.co/jockey1011/qwen2.5-coder-3b-instruct-uncensored-dolphin-gguf)
# so jockey1011/qwen2.5-coder-3b-instruct-uncensored-dolphin-gguf ( but you can't train gguf )
# username/model-id
# adapter_path = the path you have your model that you have stored and trained using finetune-llm.py
# output_dir = folder you want to output the merged model.
"""

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("Loading adapter and merging...")
model = PeftModel.from_pretrained(model, adapter_path)
model = model.merge_and_unload()

print(f"Saving merged model to {output_dir}...")
model.save_pretrained(output_dir)
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
tokenizer.save_pretrained(output_dir)

print("Done! Model merged successfully.")
