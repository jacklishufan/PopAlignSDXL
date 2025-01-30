export SDXL="stabilityai/stable-diffusion-xl-base-1.0"
export OUTPUT_DIR_DIVERSE="img/sdxl_diverse_racegender"
export OUTPUT_DIR_AGE="img/sdxl_diverse_age"
export OUTPUT_DIR_SEXUAL_ORIENTATION_DIVERSE="img/sdxl_diverse_couple"

export OUTPUT_DIR_BASE="img/sdxl_base"
export OUTPUT_DIR_SEXUAL_ORIENTATION_BASE="img/sdxl_base_couple"



ANNO_DIVERSE_SEXUAL_ORIENTATION="data/training_prompts_aug_racegender.csv"
ANNO_BASE_SEXUAL_ORIENTATION="data/training_prompts_aug_racegender.csv"
ANNO_DIVERSE_RACEGENDER="data/training_prompts_aug_racegender.csv"
ANNO_DIVERSE_AGE="data/training_prompts_aug_age.csv"
ANNO_BASE="data/training_prompts.csv"


bash scripts/eval_run.sh 8  $SDXL $OUTPUT_DIR_DIVERSE 10 $ANNO_DIVERSE_RACEGENDER 
bash scripts/eval_run.sh 8  $SDXL $OUTPUT_DIR_AGE 100 $ANNO_DIVERSE_AGE 
bash scripts/eval_run.sh 8  $SDXL $OUTPUT_DIR_SEXUAL_ORIENTATION_DIVERSE 100 $ANNO_DIVERSE_SEXUAL_ORIENTATION 


bash scripts/eval_run.sh 8  $SDXL $OUTPUT_DIR_BASE 100 $ANNO_BASE 
bash scripts/eval_run.sh 8  $SDXL $OUTPUT_DIR_SEXUAL_ORIENTATION_BASE 100 $ANNO_BASE_SEXUAL_ORIENTATION 


# run classifier in parallel
CUDA_VISIBLE_DEVICES=0 python eval/classifier.py -p $OUTPUT_DIR_DIVERSE 
CUDA_VISIBLE_DEVICES=0 python eval/classifier.py -p $OUTPUT_DIR_AGE 
CUDA_VISIBLE_DEVICES=0 python eval/classifier.py -p $OUTPUT_DIR_SEXUAL_ORIENTATION_DIVERSE
CUDA_VISIBLE_DEVICES=0 python eval/classifier.py -p $OUTPUT_DIR_BASE
CUDA_VISIBLE_DEVICES=0 python eval/classifier.py -p $OUTPUT_DIR_SEXUAL_ORIENTATION_BASE