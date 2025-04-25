# Latent Video Dataset Distillation

Hello! Thanks for checking out our [repo](https://github.com/Ning9319/Latent_Video_Dataset_Distillation)! This is official implementation of Latent Video Dataset Distillation.

## Setup
1. Clone our repo by following the command below
```
git clone https://github.com/Ning9319/Latent_Video_Dataset_Distillation.git
cd Latent_Video_Dataset_Distillation
```

2. Then install the required environments
```
pip install -r requirements.txt
```
3. Datasets preparation:
For UCF101 and HMDB51, you can use [mmaction2](https://github.com/open-mmlab/mmaction2) to extract the raw frame and then resize them using [resize_mydata.py](./distill_utils/resize_mydata.py). For Kinetics-400 and Something-Something V2, you can extract frames using the code in [extract_frames/](./extract_frames/) which were modified from [video_distillation](https://github.com/yuz1wan/video_distillation). 

4. Please download the quantized model weights from [Hugging Face](https://huggingface.co/datasets/Ning9319/Latent_Video_Dataset_Distillation/tree/main).

## Running
Below are a few example commands to get started.

### HMDB51
```
# bash main.sh GPU_num Dataset IPC
bash main.sh 0 HMDB51 24
```

### MiniUCF101
```
bash main.sh miniUCF101 24
```
Please remember to turn off the preload option in [main.sh](./main.sh) before applying it to the large-scale datasets (K400 and SSv2).

## Workflow
![overall_workflow](https://github.com/Ning9319/Latent_Video_Dataset_Distillation/blob/main/resources/method.png)


## Acknowledgements
Our code is developed based on the following codebases, thanks for sharing
* [Dancing with Still Images: Video Distillation via Static-Dynamic Disentanglement](https://github.com/yuz1wan/video_distillation)
* [CV-VAE: A Compatible Video VAE for Latent Generative Video Models](https://github.com/AILab-CVC/CV-VAE)
* [MM Action2](https://github.com/open-mmlab/mmaction2)
* [Stability AI](https://huggingface.co/stabilityai/sd-vae-ft-mse/tree/main)
