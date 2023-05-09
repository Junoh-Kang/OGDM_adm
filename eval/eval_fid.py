import os
import subprocess 
from glob import glob

os.chdir("/home/junoh/2022_DM/adm")

gpu_no = "1"
batch_size = "512"
ref_dir = "data/celeba_fid.npz"
output_file = "eval/fid/celeba_scratch.txt"

project_dirs = [
    "logs/celeba_64/00_baseline@uniform,G=0.0:2023-04-08-21-25-49-813419",
    "logs/celeba_64/06_k=0.1@G=0.01:2023-04-15-18-47-28-701755",
    # "logs/cifar10_32/00_baseline@G=0.0:2023-04-28-22-46-37-350895",
    # "logs/cifar10_32/06_k=0.1@G=0.01:2023-04-29-05-56-57-541934",
]
models = [
    "ema_0.9999_200000.pt"
    # "ema_0.9999_005000.pt",
    # "ema_0.9999_010000.pt",
    # "ema_0.9999_015000.pt",
    # "ema_0.9999_020000.pt",
    # "ema_0.9999_025000.pt",
    # "ema_0.9999_030000.pt",
]
sample_types = ["F-PNDM100", "F-PNDM50", "F-PNDM20", "F-PNDM10"]

for project_dir in project_dirs:
    for model in models:
        sample_dir = f"{project_dir}/fid/{model}"
        # log title
        subprocess.run(['/bin/bash', 'eval/0_title.sh', sample_dir, output_file])
        # samp_types = glob("f{samp_dir}/*")
        for sample_type in sample_types:
            # samp_type = samp_type.split("/")[-1]
            # log fids
            subprocess.run(['/bin/bash', 'eval/1_fid.sh', 
                            gpu_no, #1
                            batch_size, #2
                            ref_dir, #3
                            sample_dir, #4
                            sample_type, #5
                            output_file, #6
                            ])