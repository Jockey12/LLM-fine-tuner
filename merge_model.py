import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Paths
base_model_id = "GetSoloTech/Gemma3-Code-Reasoning-4B"
adapter_path = "./gemma3-code-qlora-adapter"
output_dir = "./gemma3-merged-model"

print(f"Loading base model: {base_model_id}")
# Load in CPU memory (RAM) to avoid VRAM OOM
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="cpu",
)

print(f"Loading LoRA adapter: {adapter_path}")
model = PeftModel.from_pretrained(model, adapter_path)

print("Merging weights...")
model = model.merge_and_unload()

print(f"Saving merged model to: {output_dir}")
model.save_pretrained(output_dir)

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
tokenizer.save_pretrained(output_dir)

print("Done! You can now convert 'gemma3-merged-model' to GGUF.")
