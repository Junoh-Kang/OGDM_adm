"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import yaml
import argparse
import os

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist

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

    if dist.get_rank() == 0:
        logger.configure(dir=args.log_dir, 
                         project=args.project, exp=args.exp, config=cfg)
        logger.log("creating model and diffusion...")
    
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    if dist.get_rank() == 0:
        breakpoint()
    ckpt_path = '{}/model/{}'.format(args.model_path, args.pt_name)
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    

    diffusion_kwargs = args_to_dict(args, diffusion_defaults().keys())

    for sample_type in self.sample_type:
        #sample function
        self.diffusion_kwargs['timestep_respacing'] = sample_type
        if sample_type.startswith("ddim"):
            sample_fn = create_gaussian_diffusion(**diffusion_kwargs).ddim_sample_loop
        else:
            sample_type = "ddpm" + str(sample_type)
            sample_fn = create_gaussian_diffusion(**diffusion_kwargs).p_sample_loop

    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    # save images in directory
    sample_dir = '{}/sample_from_model'.format(args.model_path)
    sample_type_dir = os.path.join(sample_dir, args.sample_type)

    logger.log("sampling...")
    all_images = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        
        sample = sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()
        
        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(sample_dir, f"samples_{args.sample_type}_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)


        for idx in range(len(arr)):
            #file_path = os.path.join(sample_type_dir, "{}.png".format{idx})
            with bf.BlobFile(sample_type_dir, "wb") as f:
                Image.fromarray(img).save(f + str(idx))


    dist.barrier()
    logger.log("sampling complete")

def load_config(cfg_dir):
    with open(cfg_dir) as f: 
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg

# def create_argparser_and_config():
#     tmp_parser = argparse.ArgumentParser()
#     tmp_parser.add_argument('--model_path', type=str)
#     tmp_parser.add_argument('--pt_name', type=str)
#     tmp_parser.add_argument('--clip_denoised', type=bool, default=True)
#     tmp_parser.add_argument('--num_samples', type=int, default=50000)
#     tmp_parser.add_argument("--local_rank", type=int) # For DDP
#     tmp_parser.add_argument("--rank", type=int) # For DDP
#     tmp_parser.add_argument("--world_size", type=int) # For DDP
#     tmp_parser.add_argument("--gpu", type=int) # For DDP
#     tmp_parser.add_argument("--dist_url", type=int) # For DDP

#     # tmp_parser.add_argument('--batch_size', type=int, default=64)
#     try:
#         tmp = load_config('./configs/_default.yaml')
#     except:
#         tmp = load_config('./configs_lg/_default.yaml')
#     tmp_args = tmp_parser.parse_args()
#     add_dict_to_argparser(tmp_parser, tmp)
    
#     cfg = load_config(f"{tmp_args.model_path}/config.yaml")
#     add_dict_to_argparser(parser, cfg)
#     args = tmp_parser.parse_args()
#     torch.cuda.set_device(args.local_rank)

#     return args, args_to_dict(args, cfg.keys())

def create_argparser_and_config():
    tmp_parser = argparse.ArgumentParser()
    tmp_parser.add_argument('--model_path', type=str)
    tmp_parser.add_argument('--pt_name', type=str)
    tmp_parser.add_argument('--clip_denoised', type=bool, default=True)
    tmp_parser.add_argument('--num_samples', type=int, default=50000)

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
    
    parser.add_argument("--local_rank", type=int) # For DDP
    parser.add_argument("--rank", type=int) # For DDP
    parser.add_argument("--world_size", type=int) # For DDP
    parser.add_argument("--gpu", type=int) # For DDP
    parser.add_argument("--dist_url", type=int) # For DDP
    parser.add_argument('--config', default=tmp_args.config, type=str)
    cfg = load_config(tmp_args.config)
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
