"""

NOTE: This cannot train a GGUF repo such as:
  https://huggingface.co/mradermacher/Gemma3-Code-Reasoning-4B-i1-GGUF

Instead, pass a non-GGUF HF base model id via --base_model_id, then fine-tune.
Afterwards you can export/quantize to GGUF.

Install:
  pip install -U "transformers>=4.44" "datasets>=2.20" "accelerate>=0.33" \
                 "peft>=0.12" "trl>=0.9" "bitsandbytes>=0.43"

Example (HF dataset):
  python finetune_gemma_code_qlora.py \
    --base_model_id <HF_BASE_MODEL_ID> \
    --dataset_name bigcode/the-stack-smol \
    --dataset_split train \
    --output_dir ./gemma3-code-qlora-adapter

Example (custom JSONL):
  python finetune_gemma_code_qlora.py \
    --base_model_id <HF_BASE_MODEL_ID> \
    --dataset_name json \
    --dataset_config_name ./train.jsonl \
    --output_dir ./gemma3-code-qlora-adapter
"""

import argparse
import os
from typing import Dict, Any

# os.environ["TORCH_BF16_IMPL"] = "0"  # disables bf16 automatic usage
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def format_example(ex: Dict[str, Any], tokenizer) -> str:
    # The Stack Smol schema: raw code in "content"
    if "content" in ex and isinstance(ex["content"], str):
        return ex["content"]

    if "messages" in ex and isinstance(ex["messages"], list):
        if hasattr(tokenizer, "apply_chat_template") and getattr(
            tokenizer, "chat_template", None
        ):
            return tokenizer.apply_chat_template(
                ex["messages"], tokenize=False, add_generation_prompt=False
            )
        out = []
        for m in ex["messages"]:
            out.append(f"{m.get('role', 'user').upper()}:\n{m.get('content', '')}\n")
        return "\n".join(out).strip() + "\n"

    if "prompt" in ex and "completion" in ex:
        return (
            f"### Instruction:\n{ex['prompt']}\n\n### Response:\n{ex['completion']}\n"
        )

    if "text" in ex:
        return str(ex["text"])

    raise ValueError(f"Unrecognized schema keys={list(ex.keys())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base_model_id", required=True, type=str, help="HF base model id (NOT GGUF)."
    )
    ap.add_argument(
        "--dataset_name", required=True, type=str, help="HF dataset name OR 'json'."
    )
    ap.add_argument(
        "--dataset_config_name",
        default=None,
        type=str,
        help="Dataset config name OR path to json/jsonl.",
    )
    ap.add_argument("--dataset_split", default="train", type=str)
    ap.add_argument("--output_dir", default="./gemma3-code-qlora-adapter", type=str)

    ap.add_argument("--max_seq_length", default=2048, type=int)
    ap.add_argument("--num_train_epochs", default=1.0, type=float)
    ap.add_argument("--per_device_train_batch_size", default=1, type=int)
    ap.add_argument("--gradient_accumulation_steps", default=16, type=int)
    ap.add_argument("--learning_rate", default=2e-4, type=float)
    ap.add_argument("--warmup_ratio", default=0.03, type=float)
    ap.add_argument("--logging_steps", default=10, type=int)
    ap.add_argument("--save_steps", default=200, type=int)
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="Override epochs with strict step count",
    )
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")

    # LoRA
    ap.add_argument("--lora_r", default=32, type=int)
    ap.add_argument("--lora_alpha", default=64, type=int)
    ap.add_argument("--lora_dropout", default=0.05, type=float)
    ap.add_argument(
        "--target_modules",
        default="q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj",
        type=str,
        help="Comma-separated module names to apply LoRA to.",
    )

    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    compute_dtype = torch.bfloat16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        # llm_int8_enable_fp32_cpu_offload=True,
    )
    # max_memory = {
    #     0: "3100MiB",  # keep some headroom on a 4GB card
    #     "cpu": "12GiB",
    # }

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        device_map={"": 0},
        # max_memory=max_memory,
        quantization_config=bnb_config,
        # torch_dtype=compute_dtype,
        dtype=compute_dtype,
    )
    # model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model.config.dtype = torch.bfloat16

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    for name, module in model.named_modules():
        if "lora_" in name:
            module.to(torch.float32)
    model.print_trainable_parameters()

    # Load dataset
    if args.dataset_name.lower() == "json":
        if not args.dataset_config_name:
            raise ValueError(
                "--dataset_config_name must be a path to JSON/JSONL when --dataset_name json"
            )
        ds = load_dataset(
            "json", data_files=args.dataset_config_name, split=args.dataset_split
        )

    else:
        if args.dataset_config_name:
            ds = load_dataset(
                args.dataset_name, args.dataset_config_name, split=args.dataset_split
            )
        else:
            ds = load_dataset(args.dataset_name, split=args.dataset_split)

    # def _map(ex):
    #     return {"text": format_example(ex, tokenizer)}
    def _map(ex):
        text = format_example(ex, tokenizer)

        # flatten maybe
        if isinstance(text, list):
            text = " ".join(text)
        # Tokenize and truncate
        tokenized = tokenizer(text, truncation=True, max_length=args.max_seq_length)
        return {
            "text": tokenizer.decode(tokenized["input_ids"], skip_special_tokens=True)
        }

    # keep_langs = {"python", "java", "javascript", "c", "cpp", "go", "php", "rust"}
    # ds = ds.filter(lambda ex: ex.get("lang") in keep_langs)
    # Keep only "text"
    ds = ds.map(_map, num_proc=os.cpu_count())
    cols_to_remove = [c for c in ds.column_names if c != "text"]
    if cols_to_remove:
        ds = ds.remove_columns(cols_to_remove)

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        weight_decay=0.0,
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        seed=args.seed,
        # 👇 THESE MOVED HERE
        max_length=args.max_seq_length,
        packing=False,
        padding_free=False,
    )
    # Old trainer
    # trainer = SFTTrainer(
    #     model=model,
    #     tokenizer=tokenizer,
    #     train_dataset=ds,
    #     dataset_text_field="text",
    #     max_seq_length=args.max_seq_length,
    #     packing=True,
    #     args=training_args,
    # )
    # New trainer setup
    trainer = SFTTrainer(
        model=model,
        train_dataset=ds,
        args=training_args,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"\nSaved LoRA adapter to: {args.output_dir}")
    print(
        "To produce a single merged model, merge the adapter into the base (see PEFT merge scripts),"
    )
    print("then convert to GGUF with llama.cpp (see export_to_gguf.py).")


if __name__ == "__main__":
    main()
