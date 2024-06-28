'''
export CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=${CUDNN_PATH}/lib
'''
from deepface import DeepFace
import glob
import argparse
import os
from tqdm.cli import tqdm
import json
import pandas as pd

def load_json(fp):
    with open(fp) as f:
        data = json.loads(f.read())
    return data

parser = argparse.ArgumentParser()
parser.add_argument('-p','--path',type=str,required=True)
args = parser.parse_args()

files = glob.glob(os.path.join(args.path,'*.jpg'))
all_rows = []
for fp in tqdm(files,position=2):
    try:
        objs = DeepFace.analyze(img_path =fp, actions = ['gender', 'race'],detector_backend = 'yunet' )
    except KeyboardInterrupt:
        exit()
    except:
        continue
    meta = load_json(fp.replace('.jpg','.json'))
    if len(objs)>0:
        meta['race'] = objs[0]['dominant_race']
        meta['gender'] = objs[0]['dominant_gender']
    else:
        meta['race'] = meta['gender'] = 'unknown'
    all_rows.append(meta)
df = pd.DataFrame(all_rows)
df.to_csv(os.path.join(args.path,'race_gender.csv'),index=False)

#df = pd.read_csv(os.path.join(args.path,'race_gender.csv'))
#df['prompt'] = df['prompt'].str.replace('Native American','NativeAmerican').str.split().apply(lambda x: ' '.join(x[3:]) if len(x) > 3 else '')
import numpy as np

def count_unique(series):
    return series.value_counts()#.to_dict()

# Group by 'prompt' and then apply the 'count_unique' function
breakpoint()
race_counts = df.groupby('prompt')['race'].apply(count_unique).to_dict()
gender_counts = df.groupby('prompt')['gender'].apply(count_unique).to_dict()
unique_prompts = df.prompt.unique()
unique_prompts = {x.split('-')[0].strip():x for x in unique_prompts}
prompt_groups = list(unique_prompts)
prompt_groups = sorted(prompt_groups)
import matplotlib.colors as mcolors
gender_balance = df.groupby('gender').count().to_dict()['prompt']
ratio_gender = np.array(list(gender_balance.values()))
race_balance = df.groupby('race').count().to_dict()['prompt']
race_balance = np.array(list(race_balance.values()))
ratio_gender = ratio_gender / ratio_gender.sum()
race_balance = race_balance  / race_balance.sum()
ratio_gender = ratio_gender -[0.5,0.5]
ratio_race = race_balance - np.ones(len(race_balance)) / len(race_balance)
discreprency_gender = np.linalg.norm(ratio_gender,ord=2)
discreprency_race = np.linalg.norm(ratio_race,ord=2)
df_res = pd.DataFrame(
    dict(
        discreprency_gender=[discreprency_gender],
        discreprency_race=[discreprency_race]
    )
)
print(df_res)
df_res.to_csv(os.path.join(args.path,'00_race_gender_summary.csv'),index=False)
