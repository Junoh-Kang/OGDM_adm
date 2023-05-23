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
    # "logs/celeba_64/00_baseline@uniform,G=0.0:2023-04-08-21-25-49-813419",
    # "logs/celeba_64/06_k=0.1@G=0.01:2023-04-15-18-47-28-701755",
    # "logs/celeba_64/07_finetune_200K@pair_T,0.15,G=0.025:2023-05-04-05-53-44-463583",
    # "logs/celeba_64/07_finetune_200K@pair_T,0.20,G=0.025:2023-05-05-01-29-18-655279",

    "logs/cifar10_32/00_baseline@G=0.0:2023-04-28-22-46-37-350895",
    # "logs/cifar10_32/06_k=0.1@G=0.01:2023-04-29-05-56-57-541934",
    # "logs/cifar10_32/07_finetune_200K@pair_T,0.15,G=0.025:2023-05-02-07-41-59-080179"
    # "logs/cifar10_32/07_finetune_200K@pair_T,0.20,G=0.025:2023-05-02-00-05-29-687760",


]
models = [
    "ema_0.9999_200000.pt",
    # "ema_0.9999_300000.pt",
    # "ema_0.9999_005000.pt",
    # "ema_0.9999_010000.pt",
    # "ema_0.9999_015000.pt",
    # "ema_0.9999_020000.pt",
    # "ema_0.9999_025000.pt",
    # "ema_0.9999_030000.pt",
]
sample_types = [
    "ddim100", "ddim50", "ddim20", "ddim10",
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