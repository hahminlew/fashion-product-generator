# Easy Finetuning Stable Diffusion
Let's easily fine-tuning pre-trained Stable Diffusion using `dataset-maker` and [LoRA](https://github.com/cloneofsimo/lora)!

## Dependencies

- python

****Please refer to [environment.yml](./environment.yml) for more details.***

```cd easy-finetuning-stable-diffusion```

```conda env create -f environment.yml```

```conda activate efsd```

## dataset-maker Instructions
`dataset-maker` is an example for a custom data collection tool to finetune Stable Diffusion. It consists of web crawler and BLIP image captioning module.

1. Inspect your desired website and slightly modify `webCrawler.py`.

2. Run `webCrawler.py`.