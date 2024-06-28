export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"

accelerate launch  --main_process_port 11149 --num_processes $1 eval/generate_images.py \
    --model_name $MODEL_NAME \
    --ckpt $2 \
    --output_dir $3 \
    --mixed_precision fp16 \
    --gen_count $4 \
    --prompts $5  ${@:6} \


