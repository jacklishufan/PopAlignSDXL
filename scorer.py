#!pip install git+https://github.com/openai/CLIP.git
#Following https://github.com/LAION-AI/aesthetic-predictor

import os
import torch
# import clip
from PIL import Image
import torch.nn as nn
from os.path import expanduser
from urllib.request import urlretrieve
import wandb_eval.aesthetic_scorer as aesthetic_lib
import wandb_eval.pick_score as pick_score
from transformers import CLIPProcessor, CLIPModel
from wandb_eval.score import build_image_reward_model,get_imagereward_score,build_clip_model,clip_score
from wandb_eval.hps import hps_initialize_model,hps_score
clip_dict = {"vit_l_14":"ViT-L/14","vit_b_32":"ViT-B/32"}

CONFIG = dict(
    aesthetic=dict(
        builder = aesthetic_lib.load_models,
        runner= aesthetic_lib.predict,
    ),
    pick=dict(
        builder = pick_score.load_model_dict,
        runner= pick_score.get_pick_score,
    ),
    image_reward=dict(
        builder=build_image_reward_model,
        runner=get_imagereward_score,
    ),
    clip=dict(
        builder=build_clip_model,
        runner=clip_score
    ),
    hps=dict(
        builder=hps_initialize_model,
        runner=hps_score
    )
)
class ScorePredictor():
    def __init__(self,device='cpu',metrics = ['aesthetic','pick']):
        self.model_dict = {x:CONFIG[x]['builder'](device=device) for x in metrics}
        self.device = device
        self.metrics = metrics
        
    def to(self,device):
        for dk,d in self.model_dict.items():
            for k,v in d.items():
                if k=='device':
                    d[k] = device
                elif isinstance(v,nn.Module):
                    d[k] = v.to(device)
            
    @torch.no_grad()        
    def __call__(self, prompt,image):
        all_scores = {}
        for m in self.metrics:
            model_dict = self.model_dict[m]
            score = CONFIG[m]['runner'](prompt=prompt, image=image,model_dict=model_dict)
            all_scores[m] = score
        return all_scores
    
    
if __name__ == '__main__':
    # test
    from diffusers.utils import load_image
    predictor = ScorePredictor(metrics=['aesthetic','pick','image_reward','clip','hps'])
    image = load_image('https://s3-us-west-2.amazonaws.com/offload-s3-dtowns-wordpress/wp-content/uploads/20171113203124/Favorite-End.jpg')
    prompt = 'a watercolor paint of Golden Gate Bridage, San Francisco'
    predictor.to('cuda')
    scores = predictor(prompt,image)
    predictor.to('cpu')
    print(scores)