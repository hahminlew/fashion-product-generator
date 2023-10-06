# Easy Finetuning Stable Diffusion
Let's easily fine-tuning pre-trained stable diffusion using dataset-maker and [LoRA](https://github.com/cloneofsimo/lora)!

## dataset-maker Instructions
`dataset-maker` consists of web crawler and BLIP image captioning module.

1. Change a 'User-Agent' variable of headers from [what-is-my-user-agent](https://www.whatismybrowser.com/detect/what-is-my-user-agent/).

***Example***
```
headers = {
    "User-Agent" : "[Copy-and-paste your user agent here]"
}
```

2. 