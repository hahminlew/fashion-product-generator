# Easy Finetuning Stable Diffusion
Let's easily fine-tuning pre-trained Stable Diffusion using `dataset-maker` and [LoRA](https://github.com/cloneofsimo/lora)!

**KREAM-Product-Displayer** is a finetuned text-to-image generative model with a custom dataset collected from [KREAM](https://kream.co.kr/), one of the best online-resell market in Korea.
Have fun creating realistic, high-quality fashion items!

## Dependencies

- python

****Please refer to [environment.yml](./environment.yml) for more details.***

```cd easy-finetuning-stable-diffusion```

```conda env create -f environment.yml```

```conda activate efsd```

## dataset-maker Instructions
`dataset-maker` is an example for a custom data collection tool to finetune Stable Diffusion. It consists of web crawler and BLIP image captioning module.

1. ```cd dataset-maker```

2. Inspect your desired website and slightly modify `webCrawler.py`.

3. Run `webCrawler.py`.

```
python webCrawler.py
```

***KREAM Product Dataset Examples Collected by dataset-maker***

<img src="./assets/examples.gif" width="100%"/>