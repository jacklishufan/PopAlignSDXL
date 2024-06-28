import pandas as pd
from itertools import product
races = ['white','asian','black','latino hispanic',"indian",'middle eastern']
genders = ['male','female']
input_csv = 'data/training_prompts.csv'
prompts = pd.read_csv(input_csv).Prompt
new_prompts = []
for p in prompts:
    for race,gender in product(races,genders):
        prompts_aug = f'a realistic photo of a {race} {gender} '+ p
        new_prompts.append(prompts_aug)
df = pd.DataFrame(new_prompts)
df.columns = ['Prompt']
df.to_csv(input_csv.replace('.csv','_aug.csv'),index=False)
# breakpoint()