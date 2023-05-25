N = 4
B = 512
sampler = "ddim100,ddim50,ddim20,ddim10"
eta=0.0
# sampler = "100,50,20,10"
project_dirs = [
    "logs/cifar10_32/07_finetune_cos_300K@pair_T,0.15,G=0.025:2023-05-25-17-11-12-258661",
    "logs/cifar10_32/07_finetune_cos_300K@pair_T,0.20,G=0.025:2023-05-25-19-12-38-739450",
    "logs/cifar10_32/07_finetune_cos_500K@pair_T,0.15,G=0.025:2023-05-25-17-12-35-848489",
    "logs/cifar10_32/07_finetune_cos_500K@pair_T,0.20,G=0.025:2023-05-25-19-14-53-409288",
]
models = [
    # "ema_0.9999_200000.pt",
    # "ema_0.9999_300000.pt",
    # "ema_0.9999_500000.pt",
    "ema_0.9999_005000.pt",
    "ema_0.9999_010000.pt",
    "ema_0.9999_015000.pt",
    # "ema_0.9999_020000.pt",
    # "ema_0.9999_025000.pt",
    # "ema_0.9999_030000.pt",
]

f = open("finetune_script.txt", 'w')
for project_dir in project_dirs:
    for model in models:   
        script = f"python -m torch.distributed.launch --nproc_per_node={N} scripts/image_sample.py " + \
                 f"--model_path {project_dir} --pt_name {model} --num_samples 50000 --batch_size {B} --sampler {sampler} --eta {eta}\n"
        # script = f"python -m torch.distributed.launch --nproc_per_node={N} scripts/image_pndm.py " + \
        #          f"--model_path {project_dir} --pt_name {model} --num_samples 50000 --batch_size {B} --sampler {sampler}\n"
        f.write(script)
f.close()

