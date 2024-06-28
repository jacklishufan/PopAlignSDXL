import pandas as pd
import numpy as np
from itertools import product
races = ['white','asian','black','latino hispanic',"indian",'middle eastern']
genders = ['male','female']


records_ref = pd.read_csv('<your folder here>/race_gender.csv')
records_good = pd.read_csv('<your folder here>/race_gender.csv')
records_good['Raw Prompt']=records_good['prompt']
for race,gender in product(races,genders):
        prompts_aug = f'a realistic photo of a {race} {gender} '
        records_good['Raw Prompt'] = records_good['Raw Prompt'].str.replace(prompts_aug,'')
# remove prefix of prompts

records_ref = records_ref.to_dict(orient='records')
unique_prompts = records_good['Raw Prompt']
records_good = records_good.to_dict(orient='records')
all_new_data = []
n_sample = 4

cap_dict = {}
raw_dict = {}

for raw_prompt in np.unique(unique_prompts):
    cap_dict[raw_prompt] = list([x for x in records_good if raw_prompt in x['Raw Prompt']])
    raw_dict[raw_prompt] = list([x for x in records_ref if raw_prompt in x['prompt']])
for _ in range(int(100*10*2*5*2)):
    raw_prompt = np.random.choice(list(cap_dict.keys()))
    if not cap_dict[raw_prompt] or not raw_dict[raw_prompt]:
        continue
    a = np.random.choice(cap_dict[raw_prompt],size=4,replace=True)
    b = np.random.choice(raw_dict[raw_prompt],size=4,replace=True)
    race_a = np.unique([x['Race'] for x in a])
    race_b = np.unique([x['Race'] for x in b])
    gender_a = np.unique([x['Gender'] for x in a])
    gender_b = np.unique([x['Gender'] for x in b])
    # sanity check
    better1 = (len(race_a) >= len(race_b)) and (len(race_a) >= len(race_b))
    if not better1:
        a,b = b,a
    for idx in range(len(a)):
        new_data = dict(
            jpg_0 = a[idx]['file_path'], 
            jpg_1 = b[idx]['file_path'],
            caption=a[idx]['Raw Prompt'],
            label=0, 
        )
        if a[idx]['gender'] == 'Woman':
            all_new_data.append(new_data)
df = pd.DataFrame(all_new_data)
print(len(np.unique(df.caption)))
df.to_csv('data/annotation_train.csv',index=False)

