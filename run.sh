#!/bin/bash

# Usage: mpiexec -n 2 --allow-run-as-root ./run.sh
# Usage2: python -m torch.distributed.launch --nproc_per_node=2 scripts/image_train.py --config configs/celeba_64/test.yaml --t_dim 1 --schedule_sampler pair,0.2
# Usage3: CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 scripts/image_train.py --config configs/celeba_64/test.yaml --t_dim 1 --schedule_sampler pair,0.2

export NCCL_DEBUG=INFO
WORLD_RANK=${OMPI_COMM_WORLD_RANK}
LOCAL_RANK=${OMPI_COMM_WORLD_LOCAL_RANK}
LOCAL_SIZE=${OMPI_COMM_WORLD_LOCAL_SIZE}

export CUDA_VISIBLE_DEVICES=${LOCAL_RANK}

python scripts/image_train.py --config configs/celeba_64/test.yaml --t_dim 1 --schedule_sampler pair,0.2

