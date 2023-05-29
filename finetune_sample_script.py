N = 4
B = 512
# sampler = "ddim50,ddim20,ddim10,ddimq50,ddimq20,ddimq10"
sampler = "ddim50,ddimq50"

eta=0.0
# sampler = "100,50,20,10"
project_dirs = [
    # "logs/cifar10_32/00_baseline@G=0.0,noise_schedule=linear:2023-05-27-02-03-59-492330",
    "logs/cifar10_32/99_rebuttal@G=0.0,noise_schedule=linear:2023-05-29-12-04-27-733346",
    # "logs/cifar10_32/99_rebuttal_dropout@G=0.0,noise_schedule=linear:2023-05-29-14-48-04-097012",
]
models = [
    # "ema_0.9999_160000.pt",
    # "ema_0.9999_170000.pt",
    "ema_0.9999_180000.pt",
    "ema_0.9999_190000.pt",
    "ema_0.9999_200000.pt",
    "ema_0.9999_210000.pt",
    # "ema_0.9999_220000.pt",
    # "ema_0.9999_230000.pt",
    # "ema_0.9999_240000.pt",

    # "ema_0.9999_005000.pt",
    # "ema_0.9999_010000.pt",
    # "ema_0.9999_015000.pt",
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

