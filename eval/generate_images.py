from guidance_scale_hack import *
from diffusers import DiffusionPipeline,UNet2DConditionModel,AutoencoderKL
import argparse
import torch
import pandas as pd
from torch.utils.data import Dataset,DataLoader
import accelerate
from tqdm.cli import tqdm
from pathlib import Path
import os
import uuid
import json
import numpy as np
torch.backends.cuda.matmul.allow_tf32 = True
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',type=str)
    parser.add_argument('--ckpt',type=str,default=None)
    parser.add_argument('--mixed_precision',type=str,default=None)
    parser.add_argument('--vae_path',type=str,default=None)
    parser.add_argument('--csft',action='store_true')
    parser.add_argument('--gen_count',default=1,type=int)
    parser.add_argument('--cfg',default=0.0,type=float)
    parser.add_argument('--guidance_scale_scheduler',default=None,type=str)
    
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    ) 
    parser.add_argument(
        "--prompts",
        type=str,
        default=None,
        help="csv path",
    ) 
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="output path",
    ) 
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="int seed",
    ) 
    parser.add_argument(
        "--lora",
        type=str,
        default='',
        help="int seed",
    ) 
    
    parser.add_argument(
        "--prefix",
        type=str,
        default='',
        help="int seed",
    ) 
    
    args = parser.parse_args()
    if args.ckpt is None:
        args.ckpt = args.model_name
    return args

class PromptDataset(Dataset):
    
    def __init__(self,path):
        self.prompts = np.unique(pd.read_csv(path).Prompt.to_list())
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, index):
        return str(self.prompts[index])
def main(args):
    dataset = PromptDataset(args.prompts)
    dataloder = DataLoader(dataset,shuffle=False,batch_size=1,collate_fn=lambda x:x)
    accelerator = accelerate.Accelerator()
    name_space = uuid.UUID('12345678123456781234567812345678')
    train_dataloader = accelerator.prepare(
     dataloder
    )
    print(f'Total Prompts: {len(train_dataloader)}/{len(dataset)}')
    vae_path = (
        args.model_name
        if args.vae_path is None
        else args.vae_path
    )
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if args.vae_path is None else None,
        revision=args.revision,
        variant=args.variant,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.ckpt, subfolder="unet", revision=args.revision, variant=args.variant
    )

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    device = accelerator.device
    pipeline = DiffusionPipeline.from_pretrained(
        args.model_name,
        vae=vae,
        unet=unet,
        revision=args.revision,
        variant=args.variant,
        safety_checker=None,
        # torch_dtype=torch.float16,
        device=device,
    ).to(device).to(weight_dtype)
    
    if args.lora:
        pipeline.load_lora_weights(args.lora)
        print("Lora Loaded!!!")
    pipeline.unet.to(memory_format=torch.channels_last)
    #pipeline.enable_xformers_memory_efficient_attention()
    if accelerator.is_local_main_process:
        Path(args.output_dir).mkdir(parents=True,exist_ok=True)
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
    for prompts in tqdm(train_dataloader):
        assert len(prompts) == 1
        prompt = prompts[0]
        if args.csft:
            actual_prompt = '[good image] '+prompt
        else:
            actual_prompt = prompt
        for gen_id in range(args.gen_count):
            prompt_uuid = uuid.uuid5(name_space,prompt)
            out_path = os.path.join(args.output_dir,f'img_{prompt_uuid}_seed_{args.seed}_{gen_id}.jpg')
            out_path_json = out_path.replace('.jpg','.json')
            json_metadata = dict(
                file_path=out_path,
                prompt=prompt
            )
            extra_args = {}
            if args.cfg:
                extra_args['guidance_scale']=args.cfg
            if args.guidance_scale_scheduler:
                extra_args['guidance_scale_scheduler'] = args.guidance_scale_scheduler
            result = pipeline(
                prompt=actual_prompt,
                num_inference_steps=50,
                **extra_args
            )
            img = result[0][0]
            img.save(out_path)
            with open(out_path_json,'w') as f:
                f.write(json.dumps(json_metadata))
                
    

if __name__ == "__main__":
    args = parse_args()
    main(args)
