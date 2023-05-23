#!/bin/sh
cd /home/junoh/2022_DM/adm

gpus=$1
batch=$2
dset1=$3
dset2=$4/$5
sample_type=$5
txt=$6

# output=$(CUDA_VISIBLE_DEVICES=$1 python -m pytorch_fid $3 $4/$5 --batch-size $2)
output=$(CUDA_VISIBLE_DEVICES=$gpus python -m pytorch_fid $dset1 $dset2 --batch-size $batch)
echo $sample_type : $output >> $txt

#1 : gpu no.
#2 : batch size
#3 : reference dir
#4 : project dir
#5 : sample type
#6 : txt

# CUDA_VISIBLE_DEVICES=0 python -m pytorch_fid --save-stats data/celeba_64_ref data/celeba_fid.npz & 
# CUDA_VISIBLE_DEVICES=1 python -m pytorch_fid --save-stats data/cifar10_32 data/cifar10_fid.npz