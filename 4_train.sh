export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
export OUTPUT_DIR="<your path here>"
DIVERSITY_V5="<your data here>"

accelerate launch --main_process_port 11122 --config_file config.json --num_processes 8 main.py \
  --resolution 1024 \
  --ckpt $REAL_VISION \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --pretrained_vae_model_name_or_path=$VAE_NAME \
  --train_data_dir=$DIVERSITY_V5 \
  --pick_split train \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="bf16" \
  --dataset_name=kashif/pickascore \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --use_8bit_adam \
  --rank=8 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --learning_rate=5e-07 --lr_scheduler="constant_with_warmup" --lr_warmup_steps=200 \
  --max_train_steps=750 \
  --checkpointing_steps=250 \
  --run_validation --validation_steps=25 \
  --seed="0" \
  --report_to="wandb" \
  --checkpoints_total_limit 4 \
  --policy gt_label_w_il \
  --dataloader dpo \
  --loss popalign \
  --positive_ratio 0.5 \
  --aesthetic_score_cutoff 7.0 \
  --beta_dpo 5000 \
  --enable_xformers_memory_efficient_attention ${@:2} 


  