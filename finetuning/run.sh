accelerate config default

huggingface-cli login

export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
export DATASET_NAME="hahminlew/kream-product-blip-captions"

CUDA_LAUNCH_BLOCKING=1 accelerate launch train_text_to_image_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE_NAME \
  --dataset_name=$DATASET_NAME --caption_column="text" \
  --resolution=1024 --random_flip \
  --train_batch_size=1 \
  --num_train_epochs=10 --checkpointing_steps=1000 \
  --learning_rate=1e-06 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --seed=42 \
  --output_dir="sdxl-kream-model-lora" \
  --validation_prompt="outer, The Nike x Balenciaga down jacket black, a photography of a black down jacket with a logo on the chest" --report_to="wandb" \
  --push_to_hub