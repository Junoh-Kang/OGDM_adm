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
    args, cfg = create_argparser_and_config()
    dist_util.setup_dist(args)

    # load model
    model = create_model(
        **args_to_dict(args, model_defaults().keys())
    ).to(dist_util.dev())
    if not args.pt_name.endswith("pt"): args.pt_name += ".pt"
    ckpt_path = f"{args.project}/model/{args.pt_name}"
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
    
    #set sampling methods
    size = [args.batch_size, 3, args.image_size, args.image_size]
    diffusion_kwargs = args_to_dict(args, diffusion_defaults().keys())

    args.sample_type = args.sampler.split(",")
    for sample_type in args.sample_type:
        if "ddim" in sample_type:
            if sample_type.startswith("ddimq"):
                diffusion_kwargs['skip_type'] = "quad"
            diffusion_kwargs['timestep_respacing'] = sample_type.replace("q", "")
            sample_fn = create_gaussian_diffusion(**diffusion_kwargs).ddim_sample_loop
        elif "PNDM" in sample_type:
            if sample_type.startswith("S-PNDM"):
                args.method = "S-PNDM"
            elif sample_type.startswith("F-PNDM"):
                args.method = "F-PNDM"
            schedule = Schedule(args, {"type":"linear",
                                        "beta_start": 0.0001,
                                        "beta_end": 0.02, 
                                        "diffusion_step": 1000}, dist_util.dev())
            sample_speed = int(sample_type[6:])
            runner = Runner(schedule=schedule,
                            model=ddp_model,
                            diffusion_step=args.diffusion_steps, 
                            sample_speed=sample_speed,
                            size=size,
                            device=dist_util.dev())
        else:
            raise NoSampler

        if dist.get_rank() == 0:
            start = time.time()
            print(f"sampling {sample_type}")
        
        num_samples = args.num_samples
        num_sampled = 0
        all_images = []
        with th.no_grad():   
            while len(all_images) * size[0] < args.num_samples:
                if "ddim" in sample_type:          
                    sample = sample_fn(
                        ddp_model, 
                        (min(size[0], num_samples - len(all_images) * size[0]), *size[1:]),
                        clip_denoised=args.clip_denoised,
                        eta=args.eta
                    )
                elif "PNDM" in sample_type:
                    sample = runner.sample_fid()

                sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
                sample = sample.contiguous()
                gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
                all_images.extend([sample.cpu() for sample in gathered_samples])
                
                if dist.get_rank() == 0:
                    print(f"created {len(all_images) * size[0]} samples...{time.time()-start:.3}s")
                    # images = th.cat(all_images)
                    # for i, image in enumerate(images):
                    #     image = image.permute(1,2,0).numpy()
                    #     Image.fromarray(image).save(f"{folder}/img{num_sampled + i}.png")
     
                # num_samples -= len(all_images) * size[0]
                # num_sampled += len(all_images) * size[0]
                # all_images = []
                dist.barrier()
            if dist.get_rank() == 0:
                folder = f"{args.project}/fid/{args.pt_name}"
                os.makedirs(folder, exist_ok=True)
                if args.eta == 0:
                    save_path = f"{folder}/{sample_type}.npz"
                else:
                    save_path = f"{folder}/{sample_type}_{args.eta}.npz"

                arr = np.concatenate(all_images, axis=0)
                arr = arr[: args.num_samples]
                np.savez(save_path, arr.transpose(0,2,3,1))
        dist.barrier()
        

def load_config(cfg_dir):
    with open(cfg_dir) as f: 
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg

def create_argparser_and_config():
    tmp_parser = argparse.ArgumentParser()
    tmp_parser.add_argument('--project', type=str)
    # tmp_parser.add_argument('--pt_name', type=str)
    tmp_parser.add_argument('--clip_denoised', type=bool, default=True)
    tmp_parser.add_argument('--num_samples', type=int, default=50000)
    tmp_parser.add_argument('--sampler', type=str)
    tmp_parser.add_argument('--eta', type=float, default=0)

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
    parser.add_argument('--project', type=str)
    # parser.add_argument('--pt_name', type=str)
    parser.add_argument('--clip_denoised', type=bool, default=True)
    parser.add_argument('--num_samples', type=int, default=50000)
    parser.add_argument('--sampler', type=str)
    parser.add_argument('--eta', type=float, default=0)
    
    parser.add_argument("--local_rank", type=int) # For DDP
    parser.add_argument("--rank", type=int) # For DDP
    parser.add_argument("--world_size", type=int) # For DDP
    parser.add_argument("--gpu", type=int) # For DDP
    parser.add_argument("--dist_url", type=int) # For DDP
    parser.add_argument('--config', default=tmp_args.config, type=str)
    cfg = load_config(f"{tmp_args.project}/config.yaml")

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
