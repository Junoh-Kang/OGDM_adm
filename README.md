# Observation-Guided Diffusion Probabilistic Models

This is the codebase for [Observation-Guided DIffusion Probabilistic Models]. This repository is based on [openai/guided-diffusion](https://github.com/openai/guided-diffusion).

# Dependencies
We share the environment of the code by docker.
```
docker pull snucvlab/lgai:adm4
```
To install all pacakges in this codebase along with their dependencies, run
```
pip install -e .
```
# Model training
```
python -m torch.distributed.launch --nproc_per_node=[N] scripts/image_train.py --config [config.yaml] --batch_size [B] --schedule_sampler pair_T,[k] --lossG_weight [gamma]

# example
python -m torch.distributed.launch --nproc_per_node=4 scripts/image_train.py --config configs/celeba_64/06_k=0.15.yaml --batch_size 32 --schedule_sampler pair_T,0.15 --lossG_weight 0.0025
```
# Model resume training 
```
python -m torch.distributed.launch --nproc_per_node=[N] scripts/image_train.py --batch_size [B] --resume_checkpoint [project_dir],[resume_step]

# example
python -m torch.distributed.launch --nproc_per_node=4 scripts/image_train.py --batch_size 32 --resume_checkpoint logs/cifar10_32/06_k=0.1@G=0.01:2023-04-29-05-56-57-541934,300000 --lr_anneal_steps 500000
```
# Model finetuning
```
python -m torch.distributed.launch --nproc_per_node=[N] scripts/image_train.py --config [config_file] --batch_size [B] --finetune [model_dir] --schedule_sampler pair_T,[k] --lossG_weight [gamma]

# example
python -m torch.distributed.launch --nproc_per_node=4 scripts/image_train.py --config configs/celeba_64/07_finetune.yaml --batch_size 32 --finetune logs/celeba_64/00_baseline@uniform,G=0.0:2023-04-08-21-25-49-813419/model/model_300000.pt --exp 07_finetune_300K --schedule_sampler pair_T,0.15 --lossG_weight 0.0025
```
# Model Sampling
```
python -m torch.distributed.launch --nproc_per_node=[N] scripts/image_sample.py --project_dir [project_dir] --pt_name [pt_name] --num_samples [num_to_sample] --batch_size [B] --sampler [samplers_seperated_by_commas] --eta [eta]

# example
python -m torch.distributed.launch --nproc_per_node=4 scripts/image_sample.py --project_dir "logs/celeba_64/00_baseline@uniform,G=0.0:2023-04-08-21-25-49-813419" --pt_name "ema_0.9999_200000.pt" --num_samples 50000 --batch_size 512 --sampler "ddim100,ddim10,ddim5" --eta 0
```