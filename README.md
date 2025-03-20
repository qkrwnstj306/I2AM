# [ICLR2025] I2AM: Interpreting Image-to-Image Latent Diffusion Models via Bi-Attribution Maps
This repository is the official implementation of [I2AM](https://openreview.net/forum?id=bBNUiErs26)


[[arXiv Paper](https://arxiv.org/abs/2312.01725)]&nbsp;
[[Project Page](https://rlawjdghek.github.io/StableVITON/)]&nbsp;

![teaser](assets/intro.jpg)&nbsp;


## Environments
```bash
git clone https://github.com/qkrwnstj306/I2AM
cd I2AM

conda create --name I2AM python=3.10 -y
conda activate I2AM

# install packages
pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117
pip install pytorch-lightning==1.5.0
pip install einops
pip install opencv-python==4.7.0.72
pip install matplotlib
pip install omegaconf
pip install albumentations
pip install transformers==4.33.2
pip install xformers==0.0.19
pip install triton==2.0.0
pip install open-clip-torch==2.19.0
pip install diffusers==0.20.2
pip install scipy==1.10.1
pip install clean-fid
pip install scikit-image
conda install -c anaconda ipython -y
```

## Weights and Data
StableVITON [checkpoint](https://kaistackr-my.sharepoint.com/:f:/g/personal/rlawjdghek_kaist_ac_kr/EjzAZHJu9MlEoKIxG4tqPr0BM_Ry20NHyNw5Sic2vItxiA?e=5mGa1c) on VITONHD <br>
You can download the VITON-HD dataset from [here](https://github.com/shadow2496/VITON-HD).<br>

## Inference

- Reference-to-Generated attribution maps: `--generated_image True --controlnet False --new_attention_map False`
- Generated-to-Reference attribution maps: `--generated_image False --controlnet True --new_attention_map False`
- Specific-Reference attribution maps ($5-$th layer): `--generated_image False --controlnet True --new_attention_map True`
- If you want to Time-, Head-Level attribution maps: `--per_time_step True` or `--per_attention_head True`
```bash
#### paired
CUDA_VISIBLE_DEVICES=0 python inference.py \
 --config_path ./configs/VITONHD.yaml \
 --batch_size 1 \
 --model_load_path ./ckpts/VITONHD.ckpt \
 --save_dir ./results \
 --data_root_dir ../VITON/dataset/zalando-hd-resized \
 --generated_image True \
 --per_time_step False \
 --per_attention_head False \
 --controlnet False \
 --new_attention_map False

#### unpaired
CUDA_VISIBLE_DEVICES=1 python inference.py \
 --config_path ./configs/VITONHD.yaml \
 --batch_size 1 \
 --model_load_path ./ckpts/VITONHD.ckpt \
 --unpair \
 --data_root_dir ../VITON/dataset/zalando-hd-resized \
 --save_dir ./results \
  --generated_image True \
 --per_time_step False \
 --per_attention_head False \
 --controlnet False \
 --new_attention_map False

**Acknowledgements** Hyeryung Jang is the corresponding author.
