#!/bin/sh
cd /home/junoh/2022_DM/adm
output=$(CUDA_VISIBLE_DEVICES=$1 python -m pytorch_fid $3 $4/$5 --batch-size $2)
echo $5 : $output >> $6

#1 : gpu no.
#2 : batch size
#3 : reference dir
#4 : project dir
#5 : sample type
#6 : txt

# #!/bin/bash
# read -p "Enter the directory of reference: " dir1
# read -p "Enter the directory of samples: " dir2
# read -p "Enter the output file: " file
# read -p "Enter the GPU number to use: " gpu


# echo $dir2 >> eval/fid/$file
# output=$(CUDA_VISIBLE_DEVICES=$gpu python -m pytorch_fid $dir1 $dir2/ddim100 --batch-size 512)
# echo ddim100 : $output >> eval/fid/$file
# output=$(CUDA_VISIBLE_DEVICES=$gpu python -m pytorch_fid $dir1 $dir2/ddim50 --batch-size 512)
# echo ddim50 : $output >> eval/fid/$file
# output=$(CUDA_VISIBLE_DEVICES=$gpu python -m pytorch_fid $dir1 $dir2/ddim20 --batch-size 512)
# echo ddim20 : $output >> eval/fid/$file
# output=$(CUDA_VISIBLE_DEVICES=$gpu python -m pytorch_fid $dir1 $dir2/ddim10 --batch-size 512)
# echo ddim10 : $output >> eval/fid/$file
# output=$(CUDA_VISIBLE_DEVICES=$gpu python -m pytorch_fid $dir1 $dir2/ddim5 --batch-size 512)
# echo ddim5 : $output >> eval/fid/$file
