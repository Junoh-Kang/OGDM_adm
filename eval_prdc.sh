#!/bin/bash
read -p "Enter the directory of reference: " dir1
read -p "Enter the directory of samples: " dir2
read -p "Enter the output file: " file
read -p "Enter the GPU number to use: " gpu


echo $dir2 >> $file
output=$(CUDA_VISIBLE_DEVICES=$gpu python -m pytorch_fid $dir1 $dir2/ddim100 --batch-size 512)
echo ddim100 : $output >> eval/fid/$file
output=$(CUDA_VISIBLE_DEVICES=$gpu python -m pytorch_fid $dir1 $dir2/ddim10 --batch-size 512)
echo ddim10 : $output >> eval/fid/$file
output=$(CUDA_VISIBLE_DEVICES=$gpu python -m pytorch_fid $dir1 $dir2/ddim5 --batch-size 512)
echo ddim5 : $output >> eval/fid/$file
