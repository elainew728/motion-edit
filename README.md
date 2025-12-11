<p align="center">
    <img src="https://github.com/elainew728/motion-edit/blob/main/gh_imgs/MotionEdit.png" width="1000"/>
<p>

# MotionEdit: Benchmarking and Learning Motion-Centric Image Editing
[![MotionEdit](https://img.shields.io/badge/Arxiv-MotionEdit-b31b1b.svg?logo=arXiv)](https://motion-edit.github.io/)
[![hf_dataset](https://img.shields.io/badge/🤗-HF_Dataset-red.svg)](https://huggingface.co/datasets/elaine1wan/MotionEdit-Bench)
[![Twitter](https://img.shields.io/badge/-Twitter@yixin_wan_-black?logo=twitter&logoColor=1D9BF0)](https://x.com/yixin_wan_?s=21&t=EqTxUZPAldbQnbhLN-CETA)
[![proj_page](https://img.shields.io/badge/Project_Page-ffcae2?style=flat-square)]([https://huggingface.co/datasets/elaine1wan/MotionEdit-Bench](https://motion-edit.github.io/)) <br><br>


# 📣 News
* **[2025/12/11]**: 🤩 We release **MotionEdit**, a novel dataset and benchmark for motion-centric image editing. Along with the dataset, we propose **MotionNFT (Motion-guided Negative-aware FineTuning)**, a post-training framework with motion alignment rewards to guide models on motion editing task.


# 🔧 Usage
## 🔍 Inferencing on *MotionEdit-Bench* with Image Editing Models
We have released our [MotionEdit-Bench](https://huggingface.co/datasets/elaine1wan/MotionEdit-Bench) on Huggingface.
In this Github Repository, we provide code that supports easy inference across open-source Image Editing models: ***Qwen-Image-Edit***, ***Flux.1 Kontext [Dev]***, ***InstructPix2Pix***, ***HQ-Edit***, ***Step1X-Edit***, ***UltraEdit***, ***MagicBrush***, and ***AnyEdit***.


### Step 1: Environment Setup
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
```
Finally, configure your own huggingface token to access restricted models by modifying `YOUR_HF_TOKEN_HERE` in `run_image_editing.py`.

### Step 2: Data Preparation
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

### Step 3: Running Inference
You can run inference on model of your choice by specifying in the arguments. For instance, here's a sample script for running inference on Qwen-Image-Edit:
```
python run_image_editing.py \
    -o "./outputs/" \
    -m "qwen-image-edit" \
    --seed 42
```
