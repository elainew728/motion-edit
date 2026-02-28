<div align="center">
    <img src="./media/MotionEdit.png" width="900"/>

# MotionEdit: Benchmarking and Learning Motion-Centric Image Editing
[![MotionEdit](https://img.shields.io/badge/Arxiv-MotionEdit-b31b1b.svg?logo=arXiv)](https://motion-edit.github.io/)
[![hf_dataset](https://img.shields.io/badge/🤗-HF_Dataset-red.svg)](https://huggingface.co/datasets/elaine1wan/MotionEdit-Bench)
[![hf_model](https://img.shields.io/badge/🤗-HF_Model-blue.svg)](https://huggingface.co/elaine1wan/motionedit)
[![Twitter](https://img.shields.io/badge/-Twitter@yixin_wan_-black?logo=twitter&logoColor=1D9BF0)](https://x.com/yixin_wan_?s=21&t=EqTxUZPAldbQnbhLN-CETA)
[![proj_page](https://img.shields.io/badge/Project_Page-ffcae2?style=flat-square)](https://motion-edit.github.io/) <br>

[Yixin Wan](https://elainew728.github.io/)<sup>1,2</sup>, [Lei Ke](https://www.kelei.site/)<sup>1</sup>, [Wenhao Yu](https://wyu97.github.io/)<sup>1</sup>, [Kai-Wei Chang](https://web.cs.ucla.edu/~kwchang/)<sup>2</sup>, [Dong Yu](https://sites.google.com/view/dongyu888/)<sup>1</sup>

<sup>1</sup>*Tencent AI, Seattle*   &nbsp;  <sup>2</sup>*University of California, Los Angeles*

<img src="./media/examples_1.gif" width="100%" alt="MotionNFT Examples 1">

</div>

# ✨ Overview
**MotionEdit** is a novel dataset and benchmark for motion-centric image editing. We also propose **MotionNFT** (Motion-guided Negative-aware FineTuning), a post-training framework with motion alignment rewards to guide models on motion image editing task.

# 📣 News
* **[2026/02/20]**: 🎉 **MotionEdit** is accepted to **CVPR 2026**! See you in Denver! 😄
* **[2026/01]** We release [MotionEdit-Bench](https://huggingface.co/datasets/elaine1wan/MotionEdit-Bench) and [MotionEdit-Train](https://huggingface.co/datasets/elaine1wan/MotionEdit-Train). Enjoy! 😁 
* **[2025/12/11]**: 🤩 We release **MotionEdit**, a novel dataset and benchmark for motion-centric image editing. Along with the dataset, we propose **MotionNFT (Motion-guided Negative-aware FineTuning)**, a post-training framework with motion alignment rewards to guide models on motion editing task.

# 🔧 Usage
## 🧱 To Start: Environment Setup
Clone this github repository and switch to the directory.

```
git clone https://github.com/elainew728/motion-edit.git
cd motion-edit
```

Create and activate the conda environment with dependencies that supports inference and training. 

> * **Note:** some models like UltraEdit requires specific dependencies on the diffusers library. Please refer to their official repository to resolve dependencies before running inference.

```
conda env create -f environment.yml
conda activate motionedit
pip install flash-attn==2.7.4.post1 --no-build-isolation
```
Finally, configure your own huggingface token to access restricted models by modifying `YOUR_HF_TOKEN_HERE` in [inference/run_image_editing.py](https://github.com/elainew728/motion-edit/tree/main/inference/run_image_editing.py).


## 🔹 Quick Single-Image Demo
If you just want to edit a single image with our MotionNFT checkpoint, place the original input image file and your text prompt (in `.txt` format, same file name as the image file) inside `examples/input_examples/`. Then, run `examples/run_inference_single.py` to inference on the input image with your prompt. 

We have prepared 3 input images from our **MotionEdit-Bench** dataset in the `examples/input_examples/` folder. Play around with them by running the following example code:
```
python examples/run_inference_single.py \
    --input_image examples/input_examples/512.jpg \
    --output_dir examples/output_examples
```
The script automatically loads `examples/input_examples/512.txt` when `--prompt` is omitted. You can still override the prompt or supply a local LoRA via `--prompt`/`--lora_path` if needed.


## 🚀 Training with MotionNFT
To run training code, first change your working directory to the train folder:
```
cd train
```

### Step 0: Data Format (Optional: If you wish to use your own dataset.)
This step is for preprocessing and formatting your own data for training. *You can safely ignore this step if you plan to use our [MotionEdit-Train](https://huggingface.co/datasets/elaine1wan/MotionEdit-Train) dataset for training.* 

Please format your training data according to the following structure. Place your `{}_metadata.jsonl` files under the folder `motionedit_data/` in the `train/` directory.

Data Folder structure:

```
- motionedit_data
  - images/
     - YOUR_IMAGE_DATA
     - ...
  - train_metadata.jsonl
  - test_metadata.jsonl
```

`train_metadata.jsonl` and `test_metadata.jsonl` format:

```
{"prompt": "PROMPT", "image": ["INPUT_IMAGE_PATH", "TARGET_IMAGE_PATH"]}
...
```

### Step 1: Deploy vLLM Reward Server
To set up the vLLM server for the MLLM feedback reward, first configure the path to your local `Qwen2.5-VL-32B-Instruct` model checkpoint by modifying `YOUR_MODEL_PATH` in [train/reward_server/reward_server.py](https://github.com/elainew728/motion-edit/tree/main/train/reward_server/reward_server.py).

Then, you can start the reward server:

```
python reward_server/reward_server.py
```

### Step 2: Configure Training

See [train/config/qwen_image_edit_nft.py](https://github.com/elainew728/motion-edit/tree/main/train/config/qwen_image_edit_nft.py) and [train/config/kontext_nft.py](https://github.com/elainew728/motion-edit/tree/main/train/config/kontext_nft.py) for available configurations.

**The default setting uses MotionEdit-Train for training**. If you hope to use your own dataset, set the following in the config file:
```
config.use_hf_dataset = False
config.dataset = # Your own dataset path
```

### Step 3: Run Training

```shell
export REWARD_SERVER=[YOUR_REWARD_SERVICE_IP_ADDR]:12341
RANK=[MACHINE_RANK]
MASTER_ADDR=[MASTER_ADDR]
MASTER_PORT=[MASTER_PORT]

accelerate launch --config_file flow_grpo/accelerate_configs/deepspeed_zero2.yaml \
    --num_machines 2 --num_processes 16 \
    --machine_rank ${RANK} --main_process_ip ${MASTER_ADDR} --main_process_port ${MASTER_PORT} \
    scripts/train_nft_qwen_image_edit.py --config config/qwen_image_edit_nft.py:qwen_motion_edit_reward 
```


## 🔍 Large-Scale Inferencing on *MotionEdit-Bench* with Image Editing Models
We have released our [MotionEdit-Bench](https://huggingface.co/datasets/elaine1wan/MotionEdit-Bench) on Huggingface.
In this Github Repository, we provide code that supports easy inference across open-source Image Editing models: ***Qwen-Image-Edit***, ***Flux.1 Kontext [Dev]***, ***InstructPix2Pix***, ***HQ-Edit***, ***Step1X-Edit***, ***UltraEdit***, ***MagicBrush***, and ***AnyEdit***.

### Step 1: Data Preparation
The inference script default to using our [MotionEdit-Bench](https://huggingface.co/datasets/elaine1wan/MotionEdit-Bench), which will download the dataset from Huggingface. You can specify a `cache_dir` for storing the cached data.

Additionally, you can construct your own dataset for inference. Please organize all input images into a folder `INPUT_FOLDER` and create a `metadata.jsonl` in the same directory. The `metadata.jsonl` file **must** at least contain entries with 2 entries: 
```
{
    "file_name": IMAGE_NAME.EXT,
    "prompt": PROMPT
}
```

Then, load your dataset by:
```
from datasets import load_dataset
dataset = load_dataset("imagefolder", data_dir=INPUT_FOLDER)
```

### Step 2: Running Inference
Use the following command to run inference on **MotionEdit-Bench** with our ***MotionNFT*** checkpoint, trained on **MotionEdit** with Qwen-Image-Edit as the base model:
```
python inference/run_image_editing.py \
    -o "./outputs/" \
    -m "motionedit" \
    --seed 42
```
Alternatively, our code supports inferencing multiple open-source image editing models. You can run inference on model of your choice by specifying in the arguments. For instance, here's a sample script for running inference on Qwen-Image-Edit:
```
python inference/run_image_editing.py \
    -o "./outputs/" \
    -m "qwen-image-edit" \
    --seed 42
```

# ✏️ Citing
Please consider citing our paper if you find our research useful. We appreciate your recognition!

```bibtex
@article{motionedit,
      title={MotionEdit: Benchmarking and Learning Motion-Centric Image Editing}, 
      author={Yixin Wan and Lei Ke and Wenhao Yu and Kai-Wei Chang and Dong Yu},
      year={2025},
      journal={arXiv preprint arXiv:2512.10284},
}
```
