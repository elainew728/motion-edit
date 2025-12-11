import torch
from safetensors.torch import load_file, save_file
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--base_dir", type=str, default='lora/')
args = parser.parse_args()


state_dict = load_file(f"{args.base_dir}/adapter_model.safetensors")

new_state_dict = {}

for key, value in state_dict.items():
    new_key = key.replace("base_model.model", "transformer")
    new_state_dict[new_key] = value


save_file(new_state_dict, f"{args.base_dir}/adapter_model_converted.safetensors")
