import os
import subprocess 
from glob import glob

os.chdir("/home/junoh/2022_DM/adm")

save_dir = "logs/qual"
nrow = 8
ncol = 6
num_grids = 20

project_dirs = [
    "logs/cifar10_32/baseline/fid/ema_0.9999_280000.pt",
    "logs/cifar10_32/ours_scratch_0.10,0.01/fid/ema_0.9999_210000.pt",
    # "logs/cifar10_32/ours_ft280k_0.20,0.025/fid/ema_0.9999_015000.pt",

    # "logs/celeba_64/baseline/fid/ema_0.9999_300000.pt",
    # "logs/celeba_64/ours_scratch_0.10,0.01/fid/ema_0.9999_300000.pt",
    # "logs/celeba_64/ours_ft300k_0.20,0.025/fid/ema_0.9999_015000.pt"

    # "logs/lsun_church/baseline@uniform:2023-06-06-22-45-56-696540/fid/ema_0.9999_250000.pt",
    # "logs/lsun_church/finetune_250K@pair_T,0.20:2023-06-10-23-11-43-833075/fid/ema_0.9999_015000.pt"
]
sample_types = [
    # "ddim10", 
    "S-PNDM9",
]

f = open("./evaluations/script.txt", 'w')
for project_dir in project_dirs:
    if "cifar10" in project_dir:
        dataset = "cifar10"
    elif "celeba" in project_dir:
        dataset = "celeba"
    elif "lsun" in project_dir:
        dataset = "lsun"
        
    if "base" in project_dir:
        model = "base"
    elif "scratch" in project_dir:
        model = "scratch"
    elif ("ft" in project_dir) or ("finetune" in project_dir):
        model = "ft"
    for sample_type in sample_types:
        image_dir = f"{project_dir}/{sample_type}.npz"
        prefix = f"{nrow}x{ncol}"
        script = f"python tools/grid_images.py {image_dir} {save_dir}/{dataset}/{model}/{sample_type} " + \
                    f"--prefix {nrow}x{ncol} " + \
                    f"--nrow {nrow} --ncol {ncol} " + \
                    f"--num_grids {num_grids}\n"
            # print(script)
        f.write(script)
f.close()