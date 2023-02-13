"""
Train a diffusion model on images.
"""
import yaml
import argparse

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop


def main():
    args, cfg = create_argparser_and_config()

    dist_util.setup_dist()
    logger.configure(dir=args.log_dir, project=args.project, exp=args.exp)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    
    model.to(dist_util.dev())
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
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
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
    tmp_parser.add_argument('--config', type=str)
    tmp = load_config('./configs/_default.yaml')
    add_dict_to_argparser(tmp_parser, tmp)
    tmp_args = tmp_parser.parse_args()

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=tmp_args.config, type=str)
    cfg = load_config(tmp_args.config)
    add_dict_to_argparser(parser, cfg)
    args = parser.parse_args()

    return args, args_to_dict(args, cfg.keys())

if __name__ == "__main__":
    main()
