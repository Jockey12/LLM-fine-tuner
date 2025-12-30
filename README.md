# LLM-fine-tuner

Fine-tune an LLM (instruction tuning / LoRA) and convert the resulting model to GGUF for use with lighter inference runtimes (llama.cpp, ggml-based runtimes, etc.).
- I tried fine-tuning on my RTX 3050, but only could fine-tune a 4b model up to 64 context tokens.
- I suggest using a cloud provider with access to at least a NVIDA T4 for 16GB vram to tune a 4b model.
- A couple of good providers are:
  - [Google Colab](https://colab.research.google.com)
  - [lightning.ai](https://lightning.ai)
  - [FMHY](https://fmhy.pages.dev/developer-tools#cloud-ides-collab)

Table of contents
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Quickstart](#quickstart)
- [Data preparation](#data-preparation)
- [Fine-tuning guide](#fine-tuning-guide)
  - [LoRA (recommended for low-resource fine-tuning)](#lora-recommended-for-low-resource-fine-tuning)
  - [Full fine-tuning (if you have resources)](#full-fine-tuning-if-you-have-resources)
- [Exporting / converting to GGUF](#exporting--converting-to-gguf)
  - [Option A — Convert from HF PyTorch checkpoint using community converters](#option-a---convert-from-hf-pytorch-checkpoint-using-community-converters)
  - [Option B — Convert from Safetensors (preferred) then to GGUF](#option-b---convert-from-safetensors-preferred-then-to-gguf)
  - [Quantization notes](#quantization-notes)
- [Example commands](#example-commands)
- [Troubleshooting & tips](#troubleshooting--tips)
- [References](#references)
- [License](#license)

Overview
--------
This repository collects utilities, documentation, and examples to:
- Fine-tune a pretrained Hugging Face-style model (instruction tuning, supervised finetuning, LoRA via PEFT).
- Save the fine-tuned model in a HF-compatible format.
- Convert the saved model to GGUF (a single-file format commonly used by ggml/llama.cpp and other lightweight runtimes) so you can run the model locally with optimized C/C++ runtimes.

Features
--------
- Guidance for preparing JSON/JSONL instruction datasets.
- Recommended approach: LoRA (using [PEFT](https://github.com/huggingface/peft)) — fast and memory-efficient.
- Export & conversion steps to GGUF for use with [llama.cpp](https://github.com/ggerganov/llama.cpp) and similar runtimes.
- Notes on tokenizer compatibility and quantization.

Requirements
------------
- Python 3.11<=
- Git
- CUDA-enabled GPU (for training) or CPU-only for small experiments (LoRA still recommended)
- Typical packages (see installation below)

Recommended Python packages
- transformers
- datasets
- accelerate
- peft
- safetensors
- torch (or torch + bitsandbytes if using 4/8-bit training)
- sentencepiece / tokenizers (depending on tokenizer)
- (optionally) bitsandbytes for 8-bit training

Quickstart
------------------
1. Clone the repo:
   ```
   git clone https://github.com/Jockey12/LLM-fine-tuner.git
   cd LLM-fine-tuner
   ```

2. Create venv and install:
   ```
   python -m venv .venv
   source .venv/bin/activate
   pip install -U pip
   pip install transformers datasets accelerate peft safetensors torch
   # Optional for 8-bit / quantized training:
   pip install bitsandbytes
   ```

3. Acquire a base model:
   - Use a Hugging Face-compatible model you are licensed to use (for example a local or HF model repo).
   - Note: some models (e.g., Llama-family) require accepting model licenses on Hugging Face before download.
     
4. Fine-tune and convert to GGUF, quantization etc..

Data preparation
----------------
- Typical supervised instruction fine-tuning format: JSONL with entries like:
  ```json
  {"instruction": "Summarize the text", "input": "Long article ...", "output": "Short summary..."}
  ```
- You can adapt to other formats; the important part is to produce input/target pairs and map them to tokenized examples.
- Small example of transforming to a single JSONL file:
  ```python
  import json
  items = [
      {"instruction":"Translate to French", "input":"Hello", "output":"Bonjour"}
  ]
  with open("data.jsonl","w") as f:
      for item in items:
          f.write(json.dumps(item) + "\n")
  ```

Fine-tuning guide
-----------------
This repo recommends LoRA (via PEFT) for most use-cases: fast, cheap, and only stores a small set of adapter weights.

LoRA (recommended for low-resource fine-tuning)
- High-level steps:
  1. Load base model and tokenizer from Hugging Face.
  2. Wrap model with PEFT LoRA config.
  3. Create a dataset (tokenized) for training.
  4. Train with Trainer or Accelerate.
  5. Save the final adapter or merged model.
- finetune-llm.py does it for you.
- 
- After training you can either:
  - Save only the adapter (PEFT adapters) and load them on top of the base model at inference time.
  - Merge the adapters into the base model and save a standalone model (useful if converting to GGUF).
  - merge_model.py does it for you.
    

Full fine-tuning (if you have resources)
- Full fine-tuning updates all model weights (large compute & memory).
- Use standard HF Trainer / accelerate with appropriate batch sizes and gradient accumulation.

Exporting / converting to GGUF
------------------------------
GGUF is a single-file format used by ggml-based runtimes (commonly used with llama.cpp and other local inference tools). The conversion process usually has two phases:
1. Ensure your model is saved in a standard HF format (PyTorch .bin or safetensors).
2. Use a converter script/tool to produce a .gguf file.

Important notes before converting:
- Tokenizer compatibility: the GGUF runtime you plan to use must support the tokenizer you export. Some converters require a separate tokenizer file or convert tokenizer into the GGUF metadata.
- Prefer saving to safetensors if possible (safer & often preferred by converters).
- If you used LoRA adapters and want a single GGUF for inference, merge adapters into the base model (instructions below).

Option A — Convert from HF PyTorch checkpoint using community converters
- Many projects provide converter scripts (see references). General flow:
  1. Save model: `model.save_pretrained("my_model")` and `tokenizer.save_pretrained("my_model")`
  2. Run converter (example placeholder):
     ```
     python convert-hf-to-gguf.py --input_dir my_model --output_file my_model.gguf
     ```
  - The exact converter name and parameters depend on the converter you use (e.g., scripts in [llama.cpp](https://github.com/ggerganov/llama.cpp) or community repos).
  - Some converters support safetensors and torch .bin formats; others prefer safetensors.

Option B — Convert from safetensors (preferred) then to GGUF
- Save model with safetensors:
  ```python
  model.save_pretrained("my_model", safe_serialization=True)
  tokenizer.save_pretrained("my_model")
  ```
- Use a converter that supports safetensors:
  ```
  python convert-safetensors-to-gguf.py --input my_model --out my_model.gguf
  ```

Quantization notes
------------------
- After conversion to gguf you can optionally quantize to reduce memory and increase inference speed (e.g., 8-bit, 4-bit formats).
- Many quantizers exist in the llama.cpp ecosystem (e.g., `quantize` tools) — quantization is often performed on the GGUF or GGML file directly.
- Quantization changes accuracy; evaluate on a dev set.

Example commands
----------------
1) Example: LoRA training via accelerate (conceptual)
```
accelerate launch train_lora.py \
  --model_name_or_path "your/base-model" \
  --data_path data.jsonl \
  --output_dir lora-output \
  --per_device_train_batch_size 1 \
  --num_train_epochs 3 \
  --learning_rate 3e-4 \
  --lora_r 8 --lora_alpha 32 --lora_dropout 0.05
```

2) Save and merge PEFT adapters into a single model (so you can convert a standalone model):
```python
# pseudocode / pattern
from peft import PeftModel, PeftConfig
# load base and adapter
base = AutoModelForCausalLM.from_pretrained("your/base-model")
adapter = PeftModel.from_pretrained(base, "lora-output")
# merge
merged = adapter.merge_and_unload()
merged.save_pretrained("merged-model", safe_serialization=True)
tokenizer.save_pretrained("merged-model")
```

3) Convert merged-model to GGUF (example placeholder)
```
# converter is not part of this repo; use a converter of your choice (e.g., from llama.cpp or community)
python /path/to/convert-to-gguf.py --model-dir merged-model --out merged-model.gguf
```

Troubleshooting & tips
----------------------
- If conversion fails, check:
  - That `config.json`, tokenizer files, and model weights are present in the saved folder.
  - Whether your converter expects `pytorch_model.bin` vs `*.safetensors`.
  - Tokenizer type compatibility (some runtimes expect `tokenizer.model` for SentencePiece).
- If inference with GGUF shows poor text generation:
  - Ensure tokenizer and special tokens are consistent with the model.
  - If you merged adapters, validate by loading the merged model with HF transformers and run a small generation test before conversion.
- Storage: GGUF files (quantized) are typically much smaller than full torch checkpoints.

References
----------
- Hugging Face Transformers: https://github.com/huggingface/transformers
- PEFT (LoRA): https://github.com/huggingface/peft
- Datasets: https://github.com/huggingface/datasets
- llama.cpp (GGML / GGUF ecosystem): https://github.com/ggerganov/llama.cpp
- safetensors: https://github.com/huggingface/safetensors

Contributing
------------
Contributions, issues, and feature requests are welcome. Please open an issue or submit a PR with clear details.

License
-------
```md
MIT License

Copyright (c) [2025-present] [Jockey12]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Acknowledgements
----------------
Thanks to the Hugging Face and ggml/llama.cpp communities for tooling and converters used across the ecosystem.
