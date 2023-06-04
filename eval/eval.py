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
    "logs/cifar10_32/baseline_280k"
    # "logs/cifar10_32/ours_scratch_0.10,0.01_210k",
]
models = [
    # "ema_0.9999_210000.pt",
    "ema_0.9999_280000.pt",

    # "ema_0.9999_005000.pt",
    # "ema_0.9999_010000.pt",
    # "ema_0.9999_015000.pt",
]
sample_types = [
    # "ddim50", "ddim20", "ddim10",
    # "ddim50_quad","ddim20_quad","ddim10_quad",
    # "ddim50_1.0", "ddim20_1.0", "ddim10_1.0",
    "F-PNDM11", "F-PNDM41"#, "F-PNDM20", "F-PNDM10"
]
fid_ref = {
    "cifar10" : "data/cifar10_fid.npz",
    "celeba" : "data/celeba_fid.npz"
}
prdc_ref = {
    "cifar10" : "data/cifar10_prdc_50k.npz",
    "celeba" : "data/celeba_prdc.npz",
}
output = {
    "cifar10" : f"eval/log{gpu_no}.txt",
    "celeba" : f"eval/log{gpu_no}.txt",
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