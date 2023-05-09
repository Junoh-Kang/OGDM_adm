# Copyright 2022 Luping Liu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import yaml
import sys
import os
import numpy as np
import torch as th

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

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    diffusion_defaults, model_defaults, discriminator_defaults,
    create_gaussian_diffusion, create_model, create_discriminator,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.pndm.runner import Runner
from guided_diffusion.pndm.schedule import Schedule

def main():
    # args, config = args_and_config()

    # if args.runner == 'sample' and config['Sample']['mpi4py']:
    #     from mpi4py import MPI

    #     comm = MPI.COMM_WORLD
    #     mpi_rank = comm.Get_rank()
    #     os.environ['CUDA_VISIBLE_DEVICES'] = str(mpi_rank)

    # device = th.device(args.device)
    args, cfg = create_argparser_and_config()
    dist_util.setup_dist(args)    

    schedule = Schedule(args, {"type":"linear",
                               "beta_start": 0.0001,
                               "beta_end": 0.02, 
                               "diffusion_step": 1000},
                               dist_util.dev())

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
    size = [args.batch_size, 3, args.image_size, args.image_size]
    
    
    for sample_speed in args.sampler.split(","):
        sample_speed = int(sample_speed)
        runner = Runner(schedule=schedule,
                        model=ddp_model,
                        diffusion_step=args.diffusion_steps, 
                        sample_speed=sample_speed,
                        size=size,
                        device=dist_util.dev()
                        )

        if dist.get_rank() == 0:
            start = time.time()
            print(f"sampling {args.method}{sample_speed}")

        folder = f"{args.model_path}/fid/{args.pt_name}/{args.method}{sample_speed}"
        os.makedirs(folder, exist_ok=True)
        num_samples = args.num_samples
        num_sampled = 0
        all_images = []
        
        with th.no_grad():
            while len(all_images) * size[0] < num_samples:                
                sample = runner.sample_fid()
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
    tmp_parser.add_argument('--sampler', type=str)
    tmp_parser.add_argument('--method', type=str, default="F-PNDM")

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
    parser.add_argument('--sampler', type=str)
    parser.add_argument('--method', type=str, default="F-PNDM")
    
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
