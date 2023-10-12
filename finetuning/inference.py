from diffusers import DiffusionPipeline
import torch
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--hub_username", default="", type=str, 
                        help="Your Hugging Face username")
    parser.add_argument("--prompt", default="The Nike x Balenciaga down jacket black.", type=str, 
                        help="Type your prompt here")
    parser.add_argument("--img_name", default="generated kream product.png", type=str, 
                        help="Save image file name") 
    parser.add_argument("--num_inference_steps", default=30, type=int, 
                        help="Number of diffusion process")
    parser.add_argument("--guidance_scale", default=7.5, type=float, 
                        help="How similar the generated image will be to the prompt")
    args = parser.parse_args()

    model_path = f"{args.hub_username}/sdxl-kream-model-lora"
    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
    pipe.to("cuda")
    pipe.load_lora_weights(model_path)

    image = pipe(args.prompt, num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale).images[0]
    image.save(args.img_name)