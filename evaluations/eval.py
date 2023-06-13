import os
import subprocess 
from glob import glob

os.chdir("/home/junoh/2022_DM/adm")

############################################################################################################
#                                                   Settings                                               #
############################################################################################################
project_dirs = [
    # "logs/cifar10_32/baseline",
    # "logs/cifar10_32/ours_scratch_0.10,0.01",
    # "logs/cifar10_32/ours_ft280k_0.10,0.025",
    # "logs/cifar10_32/ours_ft280k_0.15,0.025",
    # "logs/cifar10_32/ours_ft280k_0.20,0.025",

    # "logs/celeba_64/baseline",
    # "logs/celeba_64/ours_scratch_0.10,0.01",
    # "logs/celeba_64/ours_ft300k_0.10,0.025",
    # "logs/celeba_64/ours_ft300k_0.15,0.025",
    # "logs/celeba_64/ours_ft300k_0.20,0.025",

    "logs/lsun_church/baseline@uniform:2023-06-06-22-45-56-696540",
    "logs/lsun_church/finetune_250K@pair_T,0.20:2023-06-10-23-11-43-833075"
]
models = [
    # "ema_0.9999_210000.pt",
    # "ema_0.9999_280000.pt",
    "ema_0.9999_250000.pt",
    "ema_0.9999_015000.pt",

]
sample_types = [
    "ddim50", "ddim20", "ddim10", 
    # "ddimq50", "ddimq20", "ddimq10",
    # "S-PNDM49", "S-PNDM19", 
    # "F-PNDM41", "F-PNDM11",
    # "ddim50_1.0", "ddim20_1.0", "ddim10_1.0",
]
reference = {
    "cifar10" : "data/eval_stats/cifar10_acts.npz",
    "celeba" : "data/eval_stats/celeba_acts.npz",
    "lsun_church" : "data/eval_stats/lsun_church_acts.npz"
}
output = "evaluations/log.txt"
############################################################################################################

for project_dir in project_dirs:
    if "cifar10" in project_dir:
        dataset = "cifar10"
    if "celeba" in project_dir:
        dataset = "celeba"
    if "lsun_church" in project_dir:
        dataset = "lsun_church"
    for model in models:
        for sample_type in sample_types:
            sample = f"{project_dir}/fid/{model}/{sample_type}.npz"
            if os.path.exists(sample):
                subprocess.run(['/bin/bash', 
                                'evaluations/eval.sh', 
                                reference[dataset], 
                                sample,
                                output
                                ])