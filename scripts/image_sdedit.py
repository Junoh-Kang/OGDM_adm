"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import yaml
import argparse
import os
import time 
from PIL import Image
import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP

from guided_diffusion.image_datasets import load_data
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    diffusion_defaults, model_defaults, discriminator_defaults,
    create_gaussian_diffusion, create_model, create_discriminator,
    args_to_dict,
    add_dict_to_argparser,
)
def main():
    args, cfg = create_argparser_and_config()
    dist_util.setup_dist(args)

    # load model
    model = create_model(
        **args_to_dict(args, model_defaults().keys())
    ).to(dist_util.dev())
    if not args.pt_name.endswith("pt"): args.pt_name += ".pt"
    ckpt_path = f"{args.model_path}/model/{args.pt_name}"
    state_dict = th.load(ckpt_path)
    model.load_state_dict(state_dict)
    if th.cuda.is_available():
        use_ddp = True
        ddp_model = DDP(
            model,
            device_ids=[dist_util.dev()],
            output_device=dist_util.dev(),
            broadcast_buffers=False,
            bucket_cap_mb=128,
            find_unused_parameters=False,
        )
    else:
        use_ddp = False
        ddp_model = model
    ddp_model.eval()
    
    #get ref dir
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )
    
    #set sampling methods
    size = [args.batch_size, 3, args.image_size, args.image_size]
    diffusion_kwargs = args_to_dict(args, diffusion_defaults().keys())

    args.sample_type = args.sampler.split(",")
    for sample_type in args.sample_type:
        #sample method
        diffusion_kwargs['timestep_respacing'] = sample_type
        try:
            sample_diffusion = create_gaussian_diffusion(**diffusion_kwargs)
        except:
            continue
    
    
        #sample
        if dist.get_rank() == 0:
            start = time.time()
            print(f"sampling {sample_type}")
        
        folder = f"{args.model_path}/fid/{args.pt_name}/{args.task}/sdedit_t0={args.t}_{sample_type}"
        os.makedirs(folder, exist_ok=True)
        num_samples = args.num_samples
        num_sampled = 0
        all_images = []
        with th.no_grad():
            while len(all_images) * size[0] < num_samples:                
                batch, cond, idx = next(data)
                t0 = int(args.t * sample_diffusion.num_timesteps)
                t = th.tensor([t0] * size[0], device=dist_util.dev())
                batch_t = sample_diffusion.q_sample(batch.to(dist_util.dev()), t).clamp(-1,1)
                # breakpoint()
                sample = sample_diffusion.ddim_sample_loop(
                    model=ddp_model,
                    shape=(min(size[0], num_samples - len(all_images) * size[0]), *size[1:]),
                    clip_denoised=args.clip_denoised,
                    noise=batch_t,
                    indices=list(range(min(1,t0)))[::-1],
                )
                # breakpoint()
                
                sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
                sample = sample.contiguous()
                gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
                all_images.extend([sample.cpu() for sample in gathered_samples])
                
                if dist.get_rank() == 0:
                    print(f"created {len(all_images) * size[0]} samples...{time.time()-start:.3}s")
                    images = th.cat(all_images)
                    for i, image in enumerate(images):
                        image = image.permute(1,2,0).numpy()
                        Image.fromarray(image).save(f"{folder}/img{num_sampled + i}.png")
                        
                num_samples -= len(all_images) * size[0]
                num_sampled += len(all_images) * size[0]
                all_images = []
                dist.barrier()
        dist.barrier()
        

def load_config(cfg_dir):
    with open(cfg_dir) as f: 
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg

def create_argparser_and_config():
    tmp_parser = argparse.ArgumentParser()
    tmp_parser.add_argument('--model_path', type=str)
    tmp_parser.add_argument('--pt_name', type=str)
    tmp_parser.add_argument('--clip_denoised', type=bool, default=True)
    tmp_parser.add_argument('--num_samples', type=int, default=50000)
    tmp_parser.add_argument('--sampler', type=str, default="ddim100")
    tmp_parser.add_argument('--t', type=float, default=0.1)
    tmp_parser.add_argument('--task', type=str)

    tmp_parser.add_argument("--local_rank", type=int) # For DDP
    tmp_parser.add_argument("--rank", type=int) # For DDP
    tmp_parser.add_argument("--world_size", type=int) # For DDP
    tmp_parser.add_argument("--gpu", type=int) # For DDP
    tmp_parser.add_argument("--dist_url", type=int) # For DDP
    tmp_parser.add_argument('--config', type=str)
    try:
        tmp = load_config('./configs/_default.yaml')
    except:
        tmp = load_config('./configs_lg/_default.yaml')
    add_dict_to_argparser(tmp_parser, tmp)
    tmp_args = tmp_parser.parse_args()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--pt_name', type=str)
    parser.add_argument('--clip_denoised', type=bool, default=True)
    parser.add_argument('--num_samples', type=int, default=50000)
    parser.add_argument('--sampler', type=str, default="ddim100" )
    parser.add_argument('--t', type=float, default=0.1)
    parser.add_argument('--task', type=str)
    
    parser.add_argument("--local_rank", type=int) # For DDP
    parser.add_argument("--rank", type=int) # For DDP
    parser.add_argument("--world_size", type=int) # For DDP
    parser.add_argument("--gpu", type=int) # For DDP
    parser.add_argument("--dist_url", type=int) # For DDP
    parser.add_argument('--config', default=tmp_args.config, type=str)
    cfg = load_config(f"{tmp_args.model_path}/config.yaml")

    # check is there any omitted keys
    err = ""
    for k in tmp.keys():
        if k not in cfg.keys():
            err += k + ", "
    if err:
        err += "not implemented"       
        raise Exception(err)
    
    add_dict_to_argparser(parser, cfg)
    args = parser.parse_args()

    return args, args_to_dict(args, cfg.keys())


if __name__ == "__main__":
    main()
