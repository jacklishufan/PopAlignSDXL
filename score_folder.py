from scorer import ScorePredictor
from torch.utils.data import DataLoader,Dataset
import glob
import os
import json
from PIL import Image
import argparse
from tqdm.cli import tqdm
import accelerate
import pandas as pd

def read_json(fp):
    with open(fp) as f:
        data = json.loads(f.read())
    return data

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--path',type=str,help='image folder')
    args = parser.parse_args()
    return args
    
class FolderDataset(Dataset):
    
    def __init__(self,path) -> None:
        self.files = glob.glob(os.path.join(path,'*.json'))
        self.path = path
        super().__init__()
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        data = read_json(self.files[index])
        data['file_path'] = os.path.join(self.path,data['file_path'])
        base_name = os.path.basename(data['file_path'])
        data['image'] = Image.open(os.path.join(self.path,base_name))
        return data
    
if __name__ == '__main__':
    args = get_args()
    dataset = FolderDataset(args.path)
    metrics=['aesthetic','pick','image_reward','clip','hps']
    dataloder = DataLoader(dataset,shuffle=False,batch_size=1,collate_fn=lambda x:x)
    scorer = ScorePredictor(metrics=metrics)
    
    accelerator = accelerate.Accelerator()
    train_dataloader = accelerator.prepare(
     dataloder
    )
    scorer.to(accelerator.device)
    print(f'Total Prompts: {len(train_dataloader)}/{len(dataset)}')
    all_data = []
    for data in tqdm(train_dataloader):
        assert len(data) == 1
        prompt = data[0]['prompt']
        image =  data[0]['image']
        scores = scorer(prompt,image)
        scores['prompt'] = prompt
        scores['file_path'] = data[0]['file_path']
        all_data.append(scores)
    df = pd.DataFrame(all_data)
    local_rank = accelerator.process_index
    csv_path = os.path.join(args.path,f'eval_gpu_{local_rank}.csv')
    df.to_csv(csv_path)
    cols = df.columns
    print('----mean----')
    print(df[metrics].mean())
    print('----median----')
    print(df[metrics].median())
    df[metrics].mean().to_csv(os.path.join(args.path,f'eval_gpu_{local_rank}_mean.csv'))
    df[metrics].median().to_csv(os.path.join(args.path,f'eval_gpu_{local_rank}_median.csv'))
    # print(df_mean)
    