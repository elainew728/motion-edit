# Copyright (c) MotionEdit Team, Tencent AI Seattle (https://motion-edit.github.io/)

import argparse
from pathlib import Path
from typing import Optional
import torch
from diffusers import QwenImageEditPlusPipeline
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Single-image inference for the MotionEdit LoRA.")
    parser.add_argument("--input_image", required=True, help="Path to the input image.")
    parser.add_argument(
        "--prompt",
        default=None,
        help="Image editing instruction. If omitted, loads '<image basename>.txt' from the same folder.",
    )
    parser.add_argument(
        "--output_dir",
        default="examples/output_examples",
        help="Where the edited image will be saved.",
    )
    parser.add_argument(
        "--seed", 
        type=int, default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--output_name",
        default=None,
        help="Optional custom file name. Defaults to '<input>_motionedit.png'.",
    )
    parser.add_argument(
        "--lora_path",
        default=None,
        help="Optional local path containing the MotionEdit LoRA weights.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device used for inference. Defaults to CUDA when available.",
    )
    return parser.parse_args()


def load_motionedit_adapter(
    pipe,
    lora_path: Optional[str],
):
    if lora_path:
        print("Load lora from local: ", lora_path)
        pipe.load_lora_weights(
            lora_path,
            weight_name="adapter_model_converted.safetensors",
            adapter_name="lora",
        )

    else:
        print("Load lora from Huggingface: ", lora_path)
        pipe.load_lora_weights(
            "elaine1wan/motionedit",
            weight_name="adapter_model_converted.safetensors",
            adapter_name="lora",
        )
    pipe.set_adapters(["lora"], adapter_weights=[1])


def load_prompt(image_path: Path, override_prompt: Optional[str]) -> str:
    if override_prompt is not None:
        return override_prompt

    prompt_path = image_path.with_suffix(".txt")
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found for {image_path.name}: {prompt_path}")

    prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"Prompt file is empty: {prompt_path}")
    return prompt


def main():
    args = parse_args()

    image_path = Path(args.input_image)
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    prompt = load_prompt(image_path, args.prompt)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name or f"{image_path.stem}_motionedit.png"
    output_path = output_dir / output_name

    input_image = Image.open(image_path).convert("RGB")

    pipe = QwenImageEditPlusPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2509", torch_dtype=torch.bfloat16)
    pipe.to(args.device)
    load_motionedit_adapter(pipe, args.lora_path)

    output_image = pipe(
        num_inference_steps=28,
        image=input_image,
        prompt=prompt,
        negative_prompt=" ",
        true_cfg_scale=4.0,
        guidance_scale=1.0,
        generator=torch.Generator(device=args.device).manual_seed(args.seed),
    ).images[0]
    output_image.save(output_path)
    print(f"Saved edited image to {output_path}")


if __name__ == "__main__":
    main()
