import os
from pathlib import Path
from typing import Union, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset

import PIL
from PIL import Image
from PIL.ImageOps import exif_transpose
import argparse
from tqdm import tqdm

from transformers import CLIPProcessor, CLIPModel
import ImageReward as reward

class ImagePromptDataset(Dataset):
    """
    Dataset of prompts & images. Will read from source CSV and output full row + image
    
    Parameters:
        root (Union[str, os.PathLike]): Path to CSV of prompts and images
    """
    def __init__(self, root: Union[str, os.PathLike]):
        assert Path(root).suffix.lower() == ".csv", "Expected csv file"
        self.root = Path(root)
        self.df = pd.read_csv(root)
    
    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Tuple[str, PIL.Image.Image]:
        row = self.df.iloc[0]
        prompt = row['Prompt']
        path = row['Gen Image Path']
        
        img = Image.open(path).convert("RGB")
        img = exif_transpose(img)
        
        return (prompt, img)

@torch.no_grad()
def clip_score(dataset):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    scores = []
    for (prompt, img) in tqdm(dataset, desc='CLIP'):
        inputs = processor(text=prompt, images=img, return_tensors="pt", padding=True)
        score = model(**inputs).logits_per_image.item() # Should only be 1 value
        scores.append(score)
    
    return scores


def build_image_reward_model(device='cuda'):
    return dict(model=reward.load("ImageReward-v1.0").to(device),device=device)

@torch.no_grad()
def get_imagereward_score(prompt,image,model_dict):
    model = model_dict['model']
    score = model.score(prompt, img)
    return score

# SCORE_FNS = {"clip": clip_score,
#              "imagereward": imagereward_score}

def main(args):
    # Get data
    dataset = ImagePromptDataset(args.csv_path)
    
    output_df = dataset.df.copy()
    for score_fn_name in args.score_fns:
        scores = SCORE_FNS[score_fn_name](dataset)
        output_df[score_fn_name] = scores
    
    # Save results
    output_df.to_csv("results.csv", index=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True, 
                        help='Path to generations CSV')
    parser.add_argument('--score_fns', nargs='+', choices=list(SCORE_FNS.keys()),
                        default=list(SCORE_FNS.keys()),
                        help=f"List of score function names: {list(SCORE_FNS.keys())}\nDefault = all."
                        )
    args = parser.parse_args()
    main(args)
