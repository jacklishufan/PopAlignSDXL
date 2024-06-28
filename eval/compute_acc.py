import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sys,os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-p','--path',type=str,required=True)
args = parser.parse_args()
base_name = os.path.basename(args.path)

df = pd.read_csv(os.path.join(args.path,'race_gender.csv'))
acc_race = []
acc_gender = []
gender_map  = dict(Woman='female',Man='male')
for i,x in df.iterrows():
    acc_race.append(x.race in x.prompt)
    acc_gender.append(gender_map[x.gender] in x.prompt)
acc_race = np.array(acc_race)
acc_gender = np.array(acc_gender)
both = (acc_race&acc_gender)
n = len(acc_race)
print('acc race:',sum(acc_race) / n)
print('acc gender:',sum(acc_gender) / n)
print('acc both:',sum(both) / n)
