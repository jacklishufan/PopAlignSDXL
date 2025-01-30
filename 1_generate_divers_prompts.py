import pandas as pd
from itertools import product

## GENDER
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
df.to_csv(input_csv.replace('.csv','_aug_racegender.csv'),index=False)
# breakpoint()

## AGE
import pandas as pd
from itertools import product
# races = ['white','asian','black','latino hispanic',"indian",'middle eastern']
# genders = ['male','female']
ages = ['young','mid-aged','very old']
input_csv = 'data/training_prompts.csv'
prompts = pd.read_csv(input_csv).Prompt
new_prompts = []
for p in prompts[:100]:
    for age in ages:
        prompts_aug = f'a realistic photo of a {age} '+ p
        new_prompts.append(prompts_aug)
df = pd.DataFrame(new_prompts)
df.columns = ['Prompt']
df.to_csv(input_csv.replace('.csv','_aug_age.csv'),index=False)
# breakpoint()