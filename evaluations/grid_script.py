import os
import subprocess 
from glob import glob

os.chdir("/home/junoh/2022_DM/adm")

save_dir = "logs/qual"
nrows = "8"
num_grids = 10

project_dirs = [
    # "logs/cifar10_32/baseline/fid/ema_0.9999_280000.pt",
    "logs/cifar10_32/ours_scratch_0.10,0.01/fid/ema_0.9999_210000.pt",
    "logs/cifar10_32/ours_ft280k_0.20,0.025/fid/ema_0.9999_015000.pt",
    # "logs/celeba_64/baseline/fid/ema_0.9999_300000.pt",
    # "logs/celeba_64/ours_scratch_0.10,0.01/fid/ema_0.9999_300000.pt",
    "logs/celeba_64/ours_ft300k_0.20,0.025/fid/ema_0.9999_015000.pt"
]
sample_types = [
    "ddim10", 
    # "ddimq20", "ddimq10",
]

f = open("./evaluations/script.txt", 'w')
for project_dir in project_dirs:
    if "cifar10" in project_dir:
        dataset = "cifar10"
    elif "celeba" in project_dir:
        dataset = "celeba"
    if "base" in project_dir:
        model = "base"
    elif "scratch" in project_dir:
        model = "scratch"
    elif "ft" in project_dir:
        model = "ft"
    for sample_type in sample_types:
        image_dir = f"{project_dir}/{sample_type}.npz"
        for nrow in nrows.split(","):
            prefix = f"{nrow}x{nrow}"
            script = f"python tools/grid_images.py {image_dir} {save_dir}/{dataset}/{model}/{sample_type} " + \
                     f"--prefix {nrow}x{nrow} " + \
                     f"--nrow {nrow} " + \
                     f"--num_grids {num_grids}\n"
            # print(script)
            f.write(script)
f.close()