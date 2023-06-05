import os
import subprocess 
from glob import glob

os.chdir("/home/junoh/2022_DM/adm")

############################################################################################################
#                                                   Settings                                               #
############################################################################################################
gpu_no = "0"
FID = True
PRDC = True
project_dirs = [
    "logs/cifar10_32/baseline",
    "logs/cifar10_32/ours_scratch_0.10,0.01",
    "logs/cifar10_32/ours_ft280k_0.15,0.025",
    "logs/cifar10_32/ours_ft280k_0.20,0.025",

    # "logs/celeba_64/baseline",
    # "logs/celeba_64/ours_scratch_0.10,0.01",
    # "logs/celeba_64/ours_ft300k_0.15,0.025",
    # "logs/celeba_64/ours_ft300k_0.20,0.025",
]
models = [
    "ema_0.9999_210000.pt",
    "ema_0.9999_280000.pt",
    # "ema_0.9999_300000.pt",

    "ema_0.9999_015000.pt",
]
sample_types = [
    "ddim50", "ddim20", "ddim10", 
    "ddimq50", "ddimq20", "ddimq10",
    "S-PNDM49", "S-PNDM19", 
    "F-PNDM41", "F-PNDM11"
    ]
reference = {
    "cifar10" : "data/eval_stats/cifar10_activation.npz",
    "celeba" : "data/celeba_fid.npz"
}
output = {
    "cifar10" : f"evaluations/log.txt",
    "celeba" : f"evaluations/log.txt",
}
############################################################################################################

for project_dir in project_dirs:
    if "cifar10" in project_dir:
        dataset = "cifar10"
    if "celeba" in project_dir:
        dataset = "celeba"
    for model in models:
        for sample_type in sample_types:
            sample = f"{project_dir}/fid/{model}/{sample_type}.npz"
            if os.path.exists(sample):
                subprocess.run(['/bin/bash', 
                                'evaluations/eval.sh', 
                                reference[dataset], 
                                sample,
                                output[dataset]
                                ])