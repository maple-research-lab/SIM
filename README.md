<!-- <p align="center">
 <img src="./assets/lumina-logo.png" width="40%"/>
 <br>
</p> -->

# $\textbf{SIM}$: One-Step Diffusion Distillation through Score Implicit Matching

<div align="center">


[![SIM](https://img.shields.io/badge/Paper-SIM-2b9348.svg?logo=arXiv)](https://arxiv.org/abs/xxx.xxx)&#160;
[![Static Badge](https://img.shields.io/badge/SIM--DiT%20checkpoints-Model(0.6B)-yellow?logoColor=violet&label=%F0%9F%A4%97%20SIM-DiT%20checkpoints)](https://huggingface.co/Alpha-VLLM/Lumina-Next-SFT-diffusers)&#160;
[![Static Badge](https://img.shields.io/badge/-MIT-MIT?logoColor=%231082c3&label=Home%20Page&link=https%3A%2F%2Fgithub.com%2FAlpha-VLLM%2FLumina-T2X%2Fblob%2Fmain%2FLICENSE)](https://maple-aigc.github.io/EqDiff/)&#160;
[![weixin](https://img.shields.io/badge/-WeChat@MAPLE实验室-000000?logo=wechat&logoColor=07C160)](https://mp.weixin.qq.com/s/UefnjlCSi6YvzVe-Xu9jjQ)



</div>

![intro_large](figs/final_img.png)

<!-- [[中文版本]](./README_cn.md) -->

## 🚀 Quick Start


```bash
python inference.py \
--dit_model_path "/path/to/our_model" \
--text_enc_path /path/to/PixArt-alpha/t5-v1_1-xxl \
--vae_path /path/to/PixArt-alpha/sd-vae-ft-ema \
--prompt "a colorful painting of a beautiful landscape" \
--output_dir out-0 \
--batch 4 --seed 112 --dtype bf16 --device cuda --init_sigma 2.5
```


## 📄 Citation

```
@article{xxx,
  title={One-Step Diffusion Distillation through Score Implicit Matching},
  author={xxx},
  journal={xxx,
  year={2024}
}
```