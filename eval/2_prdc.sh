#!/bin/sh
cd /home/junoh/2022_DM/adm

gpus=$1
batch=$2
dset1=$3
dset2=$4/$5
sample_type=$5
txt=$6

output=$(CUDA_VISIBLE_DEVICES=$gpus python src/evaluate.py -metrics prdc --N 10000 --dset1_feats $dset1 --dset2 $dset2 --batch_size $batch | tail -1) 
echo $sample_type : $output >> $txt

# CUDA_VISIBLE_DEVICES=0 python src/evaluate.py -metrics prdc --N 50000 --dset1 data/cifar10_32 --real_feat_path data/cifar10_prdc.npz --dset2 data/cifar10_32 --batch_size 256
# CUDA_VISIBLE_DEVICES=1 python src/evaluate.py -metrics prdc --N 10000 --dset1 data/celeba_64_ref --real_feat_path data/celeba_prdc.npz --dset2 data/celeba_64_ref --batch_size 256



#python src/evaluate.py -metrics prdc --N 50000 --dset1 data/cifar10_32 --real_feat_path test.npz --dset2 logs/cifar10_32/99_rebuttal_onlydropout@G=0.0,noise_schedule=linear:2023-05-29-23-58-27-369423/fid/ema_0.9999_280000.pt/ddim50 --batch_size 256
