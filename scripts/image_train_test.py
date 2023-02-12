"""
Train a diffusion model on images.
"""

import argparse
import yaml

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion_from_config,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop


def main():
    
    cfg = load_and_update_cfg('./configs/celebahq_256_default.yaml')

    dist_util.setup_dist()
    logger.configure(dir=cfg['project']['log_dir'], 
                     project=cfg['project']['name'], 
                     exp=cfg['project']['exp'])

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion_from_config(cfg)
    model.to(dist_util.dev())

    schedule_sampler = create_named_schedule_sampler(cfg['diffusion']['schedule_sampler'], diffusion)

    logger.log("creating data loader...")
    data = load_data(
        data_dir=cfg['data']['data_dir'],
        batch_size=cfg['training']['batch_size'],
        image_size=cfg['data']['image_size'],
        class_cond=cfg['model']['class_cond'],
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=cfg['training']['batch_size'],
        microbatch=cfg['training']['microbatch'],
        lr=cfg['training']['lr'],
        ema_rate=cfg['training']['ema_rate'],
        log_interval=cfg['training']['log_interval'],
        save_interval=cfg['training']['save_interval'],
        resume_checkpoint=cfg['training']['use_checkpoint'],
        use_fp16=cfg['model']['use_fp16'],
        fp16_scale_growth=cfg['training']['fp16_scale_growth'],
        schedule_sampler=cfg['diffusion']['schedule_sampler'],
        weight_decay=cfg['training']['weight_decay'],
        lr_anneal_steps=cfg['training']['lr_anneal_steps'],
    ).run_loop()

def load_and_update_cfg(cfg_dir):
    with open(cfg_dir) as f: 
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    for k, v in cfg.items():
        for k1, v1 in v.items():
            if k1 in ['lr']:
                cfg[k][k1] = float(cfg[k][k1])  
    
    breakpoint()
    return cfg

if __name__ == "__main__":
    main()
