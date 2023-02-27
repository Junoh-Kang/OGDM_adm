"""
Train a diffusion model on images.
"""
import yaml
import argparse
import wandb 

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    diffusion_defaults, model_defaults, discriminator_defaults,
    create_gaussian_diffusion, create_model, create_discriminator,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop

import torch

def main():
    args, cfg = create_argparser_and_config()
    
    dist_util.setup_dist()

    logger.configure(dir=args.log_dir, 
                     project=args.project, exp=args.exp, config=cfg)
    
    if args.use_discriminator:
        logger.log("creating model, discriminator and diffusion...")
    else:
        logger.log("creating model and diffusion...")
    
    model = create_model(
        **args_to_dict(args, model_defaults().keys())
    ).to(dist_util.dev())

    if args.use_discriminator:
        discriminator = create_discriminator(
            image_size=args.image_size,
            t_dim=args.t_dim,
        ).to(dist_util.dev())
    else:
        discriminator = None
    
    diffusion_kwargs = args_to_dict(args, diffusion_defaults().keys())
    diffusion = create_gaussian_diffusion(**diffusion_kwargs)
    
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )
   
    logger.log("training...")
    TrainLoop(
        model=model,
        discriminator=discriminator,
        diffusion=diffusion,
        diffusion_kwargs=diffusion_kwargs,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr_model=args.lr_model,
        lr_disc=args.lr_disc,
        lossG_weight=args.lossG_weight,
        lossD_type=args.lossD_type,
        grad_weight=args.grad_weight,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        sample_type=args.sample_type,
        sample_num=args.sample_num,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()

def load_config(cfg_dir):
    with open(cfg_dir) as f: 
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg

def create_argparser_and_config():
    tmp_parser = argparse.ArgumentParser()
    tmp_parser.add_argument("--local_rank", type=int) # For DDP
    tmp_parser.add_argument('--config', type=str)
    try:
        tmp = load_config('./configs/_default.yaml')
    except:
        tmp = load_config('./configs_lg/_default.yaml')
    add_dict_to_argparser(tmp_parser, tmp)
    tmp_args = tmp_parser.parse_args()

    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int) # For DDP
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
    torch.cuda.set_device(args.local_rank)

    return args, args_to_dict(args, cfg.keys())

if __name__ == "__main__":
    main()
