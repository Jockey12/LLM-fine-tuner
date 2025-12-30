"""
Helper script: prints recommended commands to convert a (merged) HF model to GGUF.

Why print commands?
- GGUF conversion/quantization is typically done via llama.cpp scripts/binaries.

Prereqs:
- git clone https://github.com/ggerganov/llama.cpp
- build llama.cpp (cmake)
- have a merged HF model directory (not just LoRA adapter)

This script does NOT perform the conversion itself; it generates the commands
you should run in your shell.
"""

import argparse
import os
import textwrap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--llama_cpp_dir",
        default="./llama.cpp",
        type=str,
        help="Path to llama.cpp checkout",
    )
    ap.add_argument(
        "--hf_merged_model_dir",
        required=True,
        type=str,
        help="Path to merged HF model folder",
    )
    ap.add_argument("--out_dir", default="./gguf-out", type=str)
    ap.add_argument("--out_name", default="model.gguf", type=str)
    ap.add_argument(
        "--quant",
        default="Q4_K_M",
        type=str,
        help="Quant type (e.g. Q8_0, Q6_K, Q4_K_M, etc.)",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_fp16 = os.path.join(args.out_dir, "model-fp16.gguf")
    out_quant = os.path.join(args.out_dir, args.out_name)

    convert_py = os.path.join(args.llama_cpp_dir, "convert_hf_to_gguf.py")
    quant_bin = os.path.join(args.llama_cpp_dir, "build", "bin", "llama-quantize")

    print(
        textwrap.dedent(
            f"""
            # 1) Convert HF -> GGUF (fp16/bf16)
            python "{convert_py}" "{args.hf_merged_model_dir}" --outfile "{out_fp16}"

            # 2) Quantize GGUF
            "{quant_bin}" "{out_fp16}" "{out_quant}" {args.quant}

            # Result:
            #   {out_quant}
            """
        ).strip()
    )


if __name__ == "__main__":
    main()
