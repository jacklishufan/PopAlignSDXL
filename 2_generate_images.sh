export SDXL="stabilityai/stable-diffusion-xl-base-1.0"
export OUTPUT_DIR_DIVERSE="img/sdxl_diverse"
export OUTPUT_DIR_BASE="img/sdxl_base"
ANNO_DIVERSE="data/eval_prompts_aug.csv"
ANNO_BASE="data/eval_prompts.csv"
rm -rf $OUTPUT_DIR_DIVERSE
rm -rf $OUTPUT_DIR_BASE
bash scripts/eval_run_sd15.sh 8  $SDXL $OUTPUT_DIR_DIVERSE 10 $ANNO_DIVERSE 
bash scripts/eval_run_sd15.sh 8  $SDXL $OUTPUT_DIR_BASE 100 $ANNO_BASE 

# run classifier
CUDA_VISIBLE_DEVICES=0 python eval/classifier.py -p $OUTPUT_DIR_DIVERSE & 
CUDA_VISIBLE_DEVICES=1 python eval/classifier.py -p $OUTPUT_DIR_BASE