from diffusers import DiffusionPipeline
import torch
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--hub_user_id", default="", type=str, 
                        help="Your Hugging Face user name")
    parser.add_argument("--prompt", default="The Nike x Balenciaga Jacket Black.", type=str, 
                        help="Type your prompt here")
    parser.add_argument("--img_name", default="generated kream product.png", type=str, 
                        help="Save image file name") 
    args = parser.parse_args()

    model_path = f"{args.hub_user_id}/sdxl-kream-model-lora"
    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
    pipe.to("cuda")
    pipe.load_lora_weights(model_path)

    image = pipe(args.prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
    image.save(args.img_name)