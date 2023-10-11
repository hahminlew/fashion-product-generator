export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
export DATASET_NAME="hahminlew/kream-product-blip-captions"

huggingface-cli login

# Stable Diffusion XL finetuning with LoRA
CUDA_LAUNCH_BLOCKING=1 accelerate launch train_text_to_image_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME --caption_column="text" \
  --resolution=512 --random_flip \
  --train_batch_size=1 \
  --num_train_epochs=2 --checkpointing_steps=500 \
  --learning_rate=1e-06 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --output_dir="sd-kream-model-lora-sdxl" \
  --train_text_encoder \
  --validation_prompt="outer, The Nike x Balenciaga Jacket Black, a photography of the Nike black down jacket with a hood and a white logo" --report_to="wandb" \
  --push_to_hub

'''
# Stable Diffusion XL finetuning
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
export DATASET_NAME="hahminlew/kream-product-blip-captions"

accelerate launch train_text_to_image_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE_NAME \
  --dataset_name=$DATASET_NAME \
  --enable_xformers_memory_efficient_attention \
  --resolution=512 --center_crop --random_flip \
  --proportion_empty_prompts=0.2 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 --gradient_checkpointing \
  --max_train_steps=10000 \
  --use_8bit_adam \
  --learning_rate=1e-06 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --report_to="wandb" \
  --validation_prompt="outer, The Nike x Balenciaga Jacket Black, a photography of the Nike black down jacket with a hood and a white logo" --validation_epochs 5 \
  --checkpointing_steps=5000 \
  --output_dir="sd-kream-model-lora-sdxl" \
  --push_to_hub
'''

'''
# Stable Diffusion finetuning
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATASET_NAME="hahminlew/kream-product-blip-captions"

CUDA_LAUNCH_BLOCKING=1 accelerate launch --mixed_precision="fp16"  train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="sd-kream-model" 
'''

'''
# Stable Diffusion finetuning with multi GPUs
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATASET_NAME="hahminlew/kream-product-blip-captions"

CUDA_LAUNCH_BLOCKING=1 accelerate launch --mixed_precision="fp16" --multi_gpu  train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=15000 \ 
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="sd-kream-model" 
'''

'''
# Stable Diffusion finetuning with LoRA
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATASET_NAME="hahminlew/kream-product-blip-captions"

CUDA_LAUNCH_BLOCKING=1 accelerate launch --mixed_precision="fp16" --multi_gpu  train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=15000 \ 
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="sd-kream-model" 
'''

