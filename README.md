# Observation-Guided Diffusion Probabilistic Models
This is the codebase for [Observation-Guided DIffusion Probabilistic Models]. This repository is based on [openai/guided-diffusion](https://github.com/openai/guided-diffusion).

# Dependencies
We share the environment of the code by docker.
```
docker pull snucvlab/obsdiff:env
```

To install all pacakges in this codebase along with their dependencies, run
```
pip install -e .
```

# Model training
```
python -m torch.distributed.launch --nproc_per_node=[N] scripts/image_train.py --config [config.yaml] --batch_size [B] --schedule_sampler pair_T,[k] --lossG_weight [gamma]

# example
python -m torch.distributed.launch --nproc_per_node=4 scripts/image_train.py --config configs/celeba_64/01_scratch.yaml --batch_size 32 --schedule_sampler pair_T,0.10 --lossG_weight 0.01
```

# Model resume training
```
python -m torch.distributed.launch --nproc_per_node=[N] scripts/image_train.py --batch_size [B] --resume_checkpoint [project_dir],[resume_step]

# example
python -m torch.distributed.launch --nproc_per_node=4 scripts/image_train.py --batch_size 32 --resume_checkpoint logs/celeba_64/ours_scratch_0.10,0.01,300000 --lr_anneal_steps 500000
```

# Model finetuning
```
python -m torch.distributed.launch --nproc_per_node=[N] scripts/image_train.py --config [config_file] --batch_size [B] --finetune [model_dir] --schedule_sampler pair_T,[k] --lossG_weight [gamma]

# example
python -m torch.distributed.launch --nproc_per_node=4 scripts/image_train.py --config configs/celeba_64/02_finetune.yaml --batch_size 32 --finetune logs/celeba_64/baseline/model/ema_0.9999_300000.pt --schedule_sampler pair_T,0.20 --lossG_weight 0.025
```

# Model Sampling
```
python -m torch.distributed.launch --nproc_per_node=[N] scripts/image_sample.py --project_dir [project_dir] --pt_name [pt_name] --num_samples [num_to_sample] --batch_size [B] --sampler [samplers_seperated_by_commas] --eta [eta]

# example
python -m torch.distributed.launch --nproc_per_node=4 scripts/image_sample.py --project_dir "logs/celeba_64/ours_scratch_0.10,0.01" --pt_name "ema_0.9999_300000.pt" --num_samples 50000 --batch_size 512 --sampler "ddim50,ddim20,ddim10,S-PNDM49,S-PNDM19,F-PNDM41,F-PNDM11"
```

# Evaluation
We share the environment of the evaluation code by docker.
```
docker pull snucvlab/obsdiff:fid
```

Then, run
```
# When activations of reference is not ready, evaluate and save activations at the same time by
python scripts/fid_prdc.py [reference] [sample] -save_act_path [ref.npz]

# When activations of reference is ready, evaluate 
python scripts/fid_prdc.py [reference] [sample]
```
to evaluate.

# Download trained models
We provide trained models for CIFAR-10, and CelebA.
The downloaded files should look like this:
```
└── project
    ├── config.yaml  
    └── models 
        ├── ema_xxxxxx.pt
```

Here are links to download:

- [CIFAR-10 baseline](https://drive.google.com/file/d/1F7deiE3_hAITp-G74B4s61PWyWnKcjT5/view?usp=drive_link)
- [CIFAR-10 obsdiff(scratch)](https://drive.google.com/file/d/1F7deiE3_hAITp-G74B4s61PWyWnKcjT5/view?usp=drive_link)
- [CIFAR-10 obsdiff(ft,k=0.15)](https://drive.google.com/file/d/1wsMu1nZ9q4efzUWmmnThy0XcSuwt3Y81/view?usp=drive_link)
- [CIFAR-10 obsdiff(ft,k=0.20)](https://drive.google.com/file/d/1HzFaFIhtd10GyU_jXqZN5UQw84HC9FGZ/view?usp=drive_link)
- [CelebA baseline](https://drive.google.com/file/d/1rVF8gO7QPF5E_2fZBSywMLAKrtt3ddt3/view?usp=drive_link)
- [CelebA obsdiff(scratch)](https://drive.google.com/file/d/1giccdeqoAURBuXvg0XfXW25vj1UtaTte/view?usp=drive_link)
- [CelebA obsdiff(ft, k=0.15)](https://drive.google.com/file/d/1jop4N4Y5lwz3k4mW7I3g6s218wi3hgr7/view?usp=drive_link)
- [CelebA obsdiff(ft, k=0.20)](https://drive.google.com/file/d/11A1YubnEtNxXKPllCaBh3YmPLq6nz1gh/view?usp=drive_link)
