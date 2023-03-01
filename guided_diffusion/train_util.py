import copy
import functools
import os
import wandb
 
import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from PIL import Image
import torchvision

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler, PairSampler
from .script_util import create_gaussian_diffusion
# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0

class TrainLoop:
    def __init__(
        self,
        *,
        model,
        discriminator,
        diffusion,
        diffusion_kwargs,
        data,
        batch_size,
        microbatch,
        lr_model,
        lr_disc,
        lossG_weight,
        lossD_type,
        grad_weight,
        ema_rate,
        log_interval,
        save_interval,
        sample_type,
        sample_num,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
    ):
        self.model = model
        self.discriminator = discriminator
        self.diffusion = diffusion
        self.diffusion_kwargs = diffusion_kwargs

        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr_model = lr_model
        self.lr_disc = lr_disc
        self.lossD_type = lossD_type
        self.lossG_weight = lossG_weight
        self.grad_weight = grad_weight

        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.sample_type = sample_type
        self.sample_num = sample_num

        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()
        
        self._load_and_sync_parameters()
        self.mp_trainer_model = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )
        self.opt_model = AdamW(
            self.mp_trainer_model.master_params, lr=self.lr_model, weight_decay=self.weight_decay
        )

        if self.discriminator:
            self.mp_trainer_disc = MixedPrecisionTrainer(
                model=self.discriminator,
                use_fp16=self.use_fp16,
                fp16_scale_growth=fp16_scale_growth,
            )
            self.opt_disc = AdamW(
                self.mp_trainer_disc.master_params, lr=self.lr_disc, weight_decay=self.weight_decay
            )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer_model.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
            if self.discriminator is not None:
                self.ddp_discriminator = DDP(
                    self.discriminator,
                    device_ids=[dist_util.dev()],
                    output_device=dist_util.dev(),
                    broadcast_buffers=False,
                    bucket_cap_mb=128,
                    find_unused_parameters=False,
                )
            else:
                self.ddp_discriminator = None
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model
            self.ddp_discriminator = self.discriminator

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )
                self.discriminator.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )
        dist_util.sync_params(self.model.parameters())
        if self.discriminator:
            dist_util.sync_params(self.discriminator.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer_model.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer_model.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt_model.load_state_dict(state_dict)
        # FIXME(junoh)
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt_model.load_state_dict(state_dict)

    def run_loop(self):
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch, cond, idx = next(self.data)
            self.run_step(batch, cond)
            if self.step % self.log_interval == 0 and self.step > 0:
                if dist.get_rank() == 0:
                    self.log_step()
                    logger.dumpkvs()
                self.sample_and_show(batch.shape, sample_num=self.sample_num)
            if (self.step % self.save_interval == 0) and self.step > 0:
                self.save()
            # self.sample_and_save(batch.shape, sampel_fid_num=1000)        
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            if dist.get_rank() == 0:
                self.log_step()
                logger.dumpkvs()
            self.save()
            self.sample_and_show(batch.shape)
            # self.sample_and_save(batch.shape, sample_fid_num=1000)

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        self._update_ema()
        self._anneal_lr()
        # self.log_step()

    def forward_backward(self, batch, cond):

        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            if isinstance(self.schedule_sampler, PairSampler):
                t, weights, s = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
            else:
                t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
                s = t
            
            # compute Generation loss and backward
            self.mp_trainer_model.zero_grad()
            
            compute_losses_G = functools.partial(
                self.diffusion.training_losses_G,
                self.ddp_model,
                self.ddp_discriminator,
                micro,
                t, s,
                model_kwargs=micro_cond,
                lossD_type=self.lossD_type
            )  
            if last_batch or not self.use_ddp:
                losses = compute_losses_G()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses_G()

            if self.ddp_discriminator:
                lossG = losses["lossDM"] * weights + self.lossG_weight * losses["lossG"]
                losses["Generation"] = lossG
            else:
                lossG = losses["lossDM"] * weights
            
            self.mp_trainer_model.backward(lossG.mean())
            self.mp_trainer_model.optimize(self.opt_model)

            # compute Discrimination loss and backward
            if self.ddp_discriminator:

                self.mp_trainer_disc.zero_grad()
                compute_losses_D = functools.partial(
                    self.diffusion.training_losses_D,
                    self.ddp_model,
                    self.ddp_discriminator,
                    micro,
                    t,s,
                    model_kwargs=micro_cond,
                    lossD_type=self.lossD_type
                )
                if last_batch or not self.use_ddp:
                    losses.update(compute_losses_D())
                else:
                    with self.ddp_model.no_sync() and self.ddp_discriminator.no_sync():
                        losses.update(compute_losses_D())
                
                lossD = losses["lossD"] + self.grad_weight / 2 * losses["grad_penalty"]
                losses["Discrimination"] = lossD
                
                self.mp_trainer_disc.backward(lossD.mean())
                self.mp_trainer_disc.optimize(self.opt_disc)
       
            # log losses
            if dist.get_rank() == 0:
                wandb.log({f"Train/Samples": 
                        (self.step + self.resume_step + 1) * self.global_batch},
                        step=self.step)
                log_loss_dict(
                    self.step,
                    self.diffusion, t, {k: v * weights for k, v in losses.items()}
                )

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer_model.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        for param_group in self.opt_model.param_groups:
            param_group["lr"] = self.lr_model * (1 - frac_done)
        for param_group in self.opt_disc.param_groups:
            param_group["lr"] = self.lr_disc * (1 - frac_done)

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer_model.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model/model_{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"model/ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer_model.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            # save model opt
            logger.log(f"saving model opt...")
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"model/opt_{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt_model.state_dict(), f)
            if self.ddp_discriminator:
                # save disc 
                logger.log(f"saving discriminator...")
                with bf.BlobFile(
                    bf.join(get_blob_logdir(), f"disc/disc_{(self.step+self.resume_step):06d}.pt"), 
                    "wb",
                ) as f:
                    th.save(self.discriminator.state_dict(), f)
                #save disc opt
                logger.log(f"saving discriminator opt...")
                with bf.BlobFile(
                    bf.join(get_blob_logdir(), f"disc/opt_{(self.step+self.resume_step):06d}.pt"),
                    "wb",
                ) as f:
                    th.save(self.opt_disc.state_dict(), f)
        dist.barrier()

    def sample(self, sample_fn, model, sample_num, size):
        """
        return (sample_num, c, h, w) tensor
        """
        import time
        start = time.time()

        model.eval()
        if dist.get_rank() == 0:
            logger.log("sampling")
        all_images = []
        # print(f"start work {dist.get_rank()}")
        with th.no_grad():
            while len(all_images) * size[0] < sample_num:                
                sample = sample_fn(
                    model, 
                    (min(size[0], sample_num - len(all_images)*size[0]), *size[1:]),
                    clip_denoised=True,
                )
                sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
                sample = sample.contiguous()
                gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
                all_images.extend([sample.cpu() for sample in gathered_samples])
                if dist.get_rank() == 0:
                    logger.log(f"created {len(all_images) * size[0]} samples...{time.time()-start:.3}s")
        # print(f"finished work {dist.get_rank()}")
        dist.barrier()
        # print("after barrier")
        model.train()
        return th.cat(all_images)
    
    def sample_and_show(self, size, sample_num=64):
        grid_row = min(max(int(sample_num**0.5),4),sample_num)
        for sample_type in self.sample_type:
            #sample function
            self.diffusion_kwargs['timestep_respacing'] = sample_type
            try:
                if sample_type.startswith("ddim"):
                    sample_fn = create_gaussian_diffusion(**self.diffusion_kwargs).ddim_sample_loop
                else:
                    sample_type = "ddpm" + str(sample_type)
                    sample_fn = create_gaussian_diffusion(**self.diffusion_kwargs).p_sample_loop
            except:
                continue
            #sample
            sample = self.sample(sample_fn=sample_fn, model=self.ddp_model, 
                                 sample_num=sample_num, size=size)
            sample = torchvision.utils.make_grid(sample, grid_row).permute(1,2,0).numpy()
            # save        
            if dist.get_rank() == 0:
                filename = f"samples/model_{sample_type}_{(self.step+self.resume_step):06d}.png"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    Image.fromarray(sample).save(f)
                wandb.log({f"{sample_type}_model": wandb.Image(sample)})
            
    def sample_and_save(self, size, sample_fid_num=1000):
        for sample_type in self.sample_type:
            #sample function
            self.diffusion_kwargs['timestep_respacing'] = sample_type
            if sample_type.startswith("ddim"):
                sample_fn = create_gaussian_diffusion(**self.diffusion_kwargs).ddim_sample_loop
            else:
                sample_type = "ddpm" + str(sample_type)
                sample_fn = create_gaussian_diffusion(**self.diffusion_kwargs).p_sample_loop
            #sample
            sampled = 0
            while sampled < sample_fid_num:
                sample = self.sample(sample_fn=sample_fn, model=self.ddp_model, 
                                     sample_num=sample_num, size=size)
                sample = sample.permute(0,2,3,1).numpy()
                # save        
                if dist.get_rank() == 0:
                    foldername = f"fid/model_{sample_type}_{(self.step+self.resume_step):06d}/"
                    for img in sample:
                        with bf.BlobFile(bf.join(get_blob_logdir(), foldername), "wb") as f:
                            Image.fromarray(img).save(f + str(sampled))
                        sampled += 1
                        if sampled == sampe_fid_num:
                            return

def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()

def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None

def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None

def log_loss_dict(step, diffusion, ts, losses):
    train_key = [key for key in losses.keys() if key in \
        ["Generation", "lossDM", "lossG", "Discrimination", "lossD", "grad_penalty"]]
    supple_key = [key for key in losses.keys() if key in \
        ["real_score", "fake_score", "real_acc", "fake_acc"]]

    for key in train_key:
        wandb.log({f"Train/{key}": losses[key].mean().item()}, step=step)
        for sub_t, sub_loss in zip(ts.cpu().numpy(), losses[key].detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            wandb.log({f"TrainSupple/{key}_q{quartile}": sub_loss}, step=step)
    
    for key in supple_key:
        wandb.log({f"Eval/{key}": losses[key].mean().item()}, step=step)
        for sub_t, sub_loss in zip(ts.cpu().numpy(), losses[key].detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            wandb.log({f"EvalSupple/{key}_q{quartile}": sub_loss}, step=step)
        
    # for key, values in losses.items():
    #     # logger.logkv_mean(key, values.mean().item())
    #     wandb.log({key: values.mean().item()}, step=step)
    #     # Log the quantiles (four quartiles, in particular).
    #     for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
    #         quartile = int(4 * sub_t / diffusion.num_timesteps)
    #         # logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
    #         wandb.log({f"{key}/q{quartile}": sub_loss}, step=step)
