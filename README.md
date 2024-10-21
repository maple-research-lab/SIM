<!-- <p align="center">
 <img src="./assets/lumina-logo.png" width="40%"/>
 <br>
</p> -->

# $\textbf{SIM}$: One-Step Diffusion Distillation through Score Implicit Matching

<div align="center">

[![Static Badge](https://img.shields.io/badge/-MAPLE--Lab-MAPLE--Lab?logoColor=%231082c3&label=Home%20Page&link=https%3A%2F%2Fgithub.com%2FMAPLE_AIGC)](https://maple-aigc.github.io)&#160;
[![weixin](https://img.shields.io/badge/-WeChat@MAPLE实验室-000000?logo=wechat&logoColor=07C160)](https://mp.weixin.qq.com/s/UefnjlCSi6YvzVe-Xu9jjQ)



[![SIM](https://img.shields.io/badge/Paper-SIM-2b9348.svg?logo=arXiv)](https://arxiv.org/abs/xxx.xxx)&#160;
[![Static Badge](https://img.shields.io/badge/SIM--DiT%20checkpoints-Model(0.6B)-yellow?logoColor=violet&label=%F0%9F%A4%97%20SIM-DiT%20checkpoints)](https://huggingface.co/xxxx)&#160;
[![Static Badge](https://img.shields.io/badge/-Project%20Page-orange?logo=healthiness&logoColor=1D9BF0)](https://maple-aigc.github.io/SIM)&#160;


</div>

![intro_large](figs/final_img.png)

## Overview

This repository contains inference-only code for our work, SIM, a cutting-edge approach for distilling pre-trained diffusion models into efficient one-step generators. Unlike traditional models that require multiple sampling steps, SIM achieves high-quality sample generation without needing training samples for distillation. It effectively computes gradients for various score-based divergences, resulting in impressive performance metrics: an FID of 2.06 for unconditional generation and 1.96 for class-conditional generation on the CIFAR10 dataset. Additionally, SIM has been applied to a state-of-the-art transformer-based diffusion model for text-to-image generation, achieving an aesthetic score of 6.42 and outperforming existing one-step generators. 

## Released Models

We released two models, one for reproduce the metrics results in our paper, another one has been trained for more steps with better generation quality.

## Inference

```bash
python inference.py \
--dit_model_path "/path/to/our_model" \
--text_enc_path /path/to/PixArt-alpha/t5-v1_1-xxl \
--vae_path /path/to/PixArt-alpha/sd-vae-ft-ema \
--prompt "a colorful painting of a beautiful landscape" \
--output_dir out-0 \
--batch 4 --seed 112 --dtype bf16 --device cuda --init_sigma 2.5
```

## More Samples

xxx

## License

"One-Step Diffusion Distillation through Score Implicit Matching" is released under Affero General Public License v3.0

## Acknowledgements

Zhengyang Geng is supported by funding from the Bosch Center for AI. Zico Kolter gratefully
acknowledges Bosch’s funding for the lab.

We also acknowledge the authors of Diff-Instruct and Score-identity Distillation for their great
contributions to high-quality diffusion distillation Python code. We appreciate the authors of PixelArt-
α for making their DiT-based diffusion model public.

## 📄 Citation

```
@article{luo2024sim,
  title={One-Step Diffusion Distillation through Score Implicit Matching},
  author={Luo, Weijian and Huang, Zemin and Geng, Zhengyang and J. Zico Kolter and Qi, Guojun},
  journal={Advances in Neural Information Processing Systems},
  year={2024}
}
```
