export OUT_NEUTRAL="<your output path here>"
export OUT_SPECIFIC="<your output path here>"

bash scripts/eval_run.sh 8  "<your model path here>" $OUT_NEUTRAL 100 data/eval_prompt_v1_neutral.csv
bash scripts/eval_run.sh 8  "<your model path here>" $OUT_SPECIFIC 100  data/eval_prompt_v1_specific.csv

# get scores


CUDA_VISIBLE_DEVICES=0 python eval/classifier.py -p $OUT_NEUTRAL
CUDA_VISIBLE_DEVICES=0 python eval/classifier.py -p $OUT_SPECIFIC

# get image quality scores
accelerate launch --main_process_port=12343 --num_processes $1 score_folder.py\
    -p $OUT_NEUTRAL
    
accelerate launch --main_process_port=12343 --num_processes $1 score_folder.py\
    -p  $OUT_SPECIFIC