# Copyright (c) MotionEdit Team, Tencent AI Seattle (https://motion-edit.github.io/)

import os
import PIL
import torch
from tqdm import tqdm
import argparse
from huggingface_hub import login
import warnings
import ray
from datasets import load_dataset

HF_TOKEN = YOUR_HF_TOKEN_HERE
login(token=HF_TOKEN)

class MagicBrush():
    def __init__(self, device=None, weight="vinesmsuic/magicbrush-jul7"):
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                        weight, 
                        torch_dtype=torch.float16
                    ).to(device)
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)
        
    def infer_one_image(self, src_image, instruct_prompt, seed):
        generator = torch.manual_seed(seed)
        image = self.pipe(instruct_prompt, image=src_image, num_inference_steps=20, image_guidance_scale=1.5, guidance_scale=7, generator=generator).images[0]
        return image


@ray.remote(num_gpus=1)
def process_slice(slice_items, model_name, lora_path, output_dir, seed):
    if model_name in ["instructpix2pix", "hqedit", "flux", "step1x", "ultraedit", "anyedit", "qwen-image-edit", "motionedit"]:
        if model_name == "instructpix2pix":
            from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionInstructPix2PixPipeline
            model_id = "timbrooks/instruct-pix2pix"
            pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        elif model_name == "hqedit":
            from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionInstructPix2PixPipeline
            model_id = "MudeHui/HQ-Edit"
            pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        elif model_name == "flux": 
            from diffusers import FluxKontextPipeline
            pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16)
        elif model_name == "step1x":
            warnings.warn("Step1X requires a special diffusers branch. Please refer to the installation command below or the original Step1X repository.")
            '''
            git clone -b step1xedit https://github.com/Peyton-Chen/diffusers.git
            cd diffusers
            pip install -e .
            '''
            from diffusers import Step1XEditPipeline
            pipe = Step1XEditPipeline.from_pretrained("stepfun-ai/Step1X-Edit-v1p1-diffusers", torch_dtype=torch.bfloat16)
        elif model_name == "ultraedit":
            warnings.warn("UltraEdit requires a special diffusers branch. Please refer to the installation command below or the original Step1X repository.")
            '''
            git clone https://github.com/HaozheZhao/UltraEdit.git
            cd diffusers
            pip install -e .
            '''
            from diffusers import StableDiffusion3InstructPix2PixPipeline
            pipe = StableDiffusion3InstructPix2PixPipeline.from_pretrained("BleachNick/SD3_UltraEdit_w_mask", torch_dtype=torch.float16)
        elif model_name in ["qwen-image-edit", "motionedit"]:
            from diffusers import QwenImageEditPlusPipeline
            pipe = QwenImageEditPlusPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2509", torch_dtype=torch.bfloat16)
        elif model_name == "anyedit":
            warnings.warn("AnyEdit requires a special model pipeline with selected adapters. Please clone the official repository and resolve dependencies before running the inference code.")
            import sys
            sys.path.append(os.getcwd())
            from AnySD.anysd.src.model import AnySDPipeline
            from AnySD.anysd.src.utils import get_experts_dir
            expert_file_path = get_experts_dir(repo_id="WeiChow/AnySD")
            task_embs_checkpoints = expert_file_path + "task_embs.bin"
            adapter_checkpoints = {
                "global": expert_file_path + "global.bin",
            }

            pipe = AnySDPipeline(adapters_list=[adapter_checkpoints], task_embs_checkpoints=task_embs_checkpoints)
        
        if model_name != "anyedit":
            pipe.to("cuda")
        print('Loaded model weights...')
    elif model_name == 'magicbrush':
        from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionInstructPix2PixPipeline
        model = MagicBrush(device="cuda")
    
    if model_name == "motionedit":
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

    for item in tqdm(slice_items):
        try:
            prompt = item["prompt"]
            key = item["id"]
            output_filename = f"{key}.jpg"
            output_filepath = os.path.join(output_dir, output_filename)
            if os.path.exists(output_filepath):
                continue

            input_image = item["input_image"]
            w, h = input_image.size

            if model_name in ["qwen-image-edit", "motionedit"]:
                output_image = pipe(
                    num_inference_steps=28,
                    image=input_image,
                    prompt=prompt,
                    negative_prompt=" ",
                    true_cfg_scale=4.0,
                    guidance_scale=1.0,
                    generator=torch.Generator(device="cuda").manual_seed(seed),
                ).images[0]
            elif model_name == "flux":
                output_image = pipe(
                    num_inference_steps=28,
                    image=input_image,
                    prompt=prompt,
                    height=h,
                    width=w,
                    generator=torch.Generator(device="cuda").manual_seed(seed),
                ).images[0]
            elif model_name == "step1x":
                output_image = pipe(
                    image=input_image,
                    prompt=prompt,
                    height=h,
                    width=w,
                    num_inference_steps=28,
                    true_cfg_scale=6.0,
                    generator=torch.Generator(device="cuda").manual_seed(seed),
                ).images[0]
            elif model_name == "ultraedit":
                input_image = input_image.resize((512, 512))
                mask_img = PIL.Image.new("RGB", input_image.size, (255, 255, 255))
                output_image = pipe(
                    prompt,
                    image=input_image,
                    mask_img=mask_img,
                    negative_prompt="",
                    num_inference_steps=50,
                    image_guidance_scale=1.5,
                    guidance_scale=7.5,
                ).images[0]
                output_image = output_image.resize((w,h))
            elif model_name == "anyedit": # TODO: Run
                output_image = pipe(
                    prompt=prompt,
                    original_image=input_image,
                    guidance_scale=3,
                    num_inference_steps=100,
                    original_image_guidance_scale=3,
                    adapter_name="general",
                )[0]
            elif model_name == "magicbrush":
                output_image = model.infer_one_image(input_image, prompt, 42)
            else:
                output_image = pipe(
                        image=input_image,
                        prompt=prompt,
                        height=h,
                        width=w,
                        guidance_scale=3.5,
                        num_inference_steps=28, 
                    ).images[0]
            
            output_image.save(output_filepath)
        except Exception as e:
            continue
  
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_path", default="./test/")
    parser.add_argument("-m", "--model", default="qwen-image-edit", help="Model to use for image editing")
    parser.add_argument("--input_path", required=False, help="Path to your own dataset. It should be an image folder with metadata.jsonl file.")
    parser.add_argument("--lora_path", type=str, required=False, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test", action='store_true')
    args = parser.parse_args()
    
    ray.init(include_dashboard=False, logging_level="ERROR")
    gpu_count = int(ray.available_resources().get("GPU", 1))
    output_path = os.path.join(args.output_path, args.model)
    os.makedirs(output_path, exist_ok=True)
    
    if args.input_path:
        dataset = load_dataset("imagefolder", data_dir=args.input_path)["train"]
    else:
        dataset = load_dataset("elaine1wan/MotionEdit-Bench")["train"]

    all_items = [row for row in tqdm(dataset)]
    if args.test:
        all_items = all_items[:10]

    slices = [all_items[i::gpu_count] for i in range(gpu_count)]
    del dataset
    del all_items

    ray.get([
        process_slice.remote(
            slices[i],
            args.model,
            args.lora_path,
            output_path,
            args.seed,
        ) for i in range(gpu_count)
    ])
   
