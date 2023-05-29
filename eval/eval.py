import os
import subprocess 
from glob import glob

os.chdir("/home/junoh/2022_DM/adm")

############################################################################################################
#                                                   Settings                                               #
############################################################################################################
gpu_no = "0"
FID = True
PRDC = False
project_dirs = [
    # "logs/cifar10_32/00_baseline@G=0.0,noise_schedule=linear:2023-05-23-18-07-35-856180",
    # "logs/cifar10_32/00_baseline@G=0.0,noise_schedule=linear:2023-05-27-02-03-59-492330",
    "logs/celeba_64/00_baseline@uniform,G=0.0:2023-04-08-21-25-49-813419",
    "logs/celeba_64/06_k=0.1@G=0.01:2023-04-15-18-47-28-701755",
]
models = [
    # "ema_0.9999_180000.pt",
    # "ema_0.9999_190000.pt",
    # "ema_0.9999_200000.pt",
    # "ema_0.9999_210000.pt",
    "ema_0.9999_300000.pt"
    # "ema_0.9999_005000.pt",
    # "ema_0.9999_010000.pt",
    # "ema_0.9999_015000.pt",
    # "ema_0.9999_020000.pt",
    # "ema_0.9999_025000.pt",
    # "ema_0.9999_030000.pt",
]
sample_types = [
    # "ddim100_quad", "ddim50_quad","ddim20_quad","ddim10_quad",
    # "ddim50", "ddim50_quad",
    "ddim100", "ddim50", "ddim20", "ddim10",
    # "ddim100_1.0", "ddim50_1.0", "ddim20_1.0", "ddim10_1.0",
    # "F-PNDM100", "F-PNDM50", "F-PNDM20", "F-PNDM10"
]
fid_ref = {
    "cifar10" : "data/cifar10_fid.npz",
    "celeba" : "data/celeba_fid.npz"
}
prdc_ref = {
    "cifar10" : "data/cifar10_prdc.npz",
    "celeba" : "data/celeba_prdc.npz",
}
output = {
    "cifar10" : "eval/log.txt",
    "celeba" : "eval/log.txt"
}
############################################################################################################



for project_dir in project_dirs:
    if "cifar10" in project_dir:
        dataset = "cifar10"
    if "celeba" in project_dir:
        dataset = "celeba"
    
    for model in models:
        sample_dir = f"{project_dir}/fid/{model}"
        # log title
        subprocess.run(['/bin/bash', 'eval/0_title.sh', sample_dir, output[dataset]])
        for sample_type in sample_types:
            if FID:
                subprocess.run(['/bin/bash', 'eval/1_fid.sh', 
                                "0", #gpu_no
                                "512", #batch_size
                                fid_ref[dataset], #ref_dir
                                sample_dir, #sample_dir
                                sample_type, #sample_type
                                output[dataset], #output_file
                                ])
            if PRDC:
                subprocess.run(['/bin/bash', 'eval/2_prdc.sh', 
                            "0", #gpu_no
                            "256", #batch_size
                            prdc_ref[dataset], #ref_dir
                            sample_dir, #sample_dir
                            sample_type, #sample_type
                            output[dataset], #output_file
                            ])