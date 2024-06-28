export SDXL="stabilityai/stable-diffusion-xl-base-1.0"
export OUTPUT_DIR_DIVERSE="your path here"
export OUTPUT_DIR_BASE="your path here"

ANNO_DIVERSE="data/training_prompts_aug.csv"
ANNO_BASE="data/training_prompts.csv"

bash scripts/eval_run.sh 8  $SDXL $OUTPUT_DIR_DIVERSE 10 $ANNO_DIVERSE 
bash scripts/eval_run.sh 8  $SDXL $OUTPUT_DIR_BASE 10 $ANNO_BASE 

# run classifier
CUDA_VISIBLE_DEVICES=0 python eval/classifier.py -p $ANNO_DIVERSE
CUDA_VISIBLE_DEVICES=0 python eval/classifier.py -p $ANNO_BASE