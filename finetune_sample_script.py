N = 4
B = 512
# sampler = "ddim50,ddimq50"
sampler = "ddimq50,ddimq20,ddimq10"
eta=0.0
# sampler = "100,50,20,10"
project_dirs = [
    # "logs/cifar10_32/00_baseline@G=0.0,noise_schedule=linear:2023-05-27-02-03-59-492330",
    # "logs/cifar10_32/99_rebuttal_onlydropout@G=0.0,noise_schedule=linear:2023-05-29-23-58-27-369423",
    # "logs/cifar10_32/99_rebuttal_onlydropout_ours@G=0.01:2023-05-30-15-25-19-884822",
    "logs/cifar10_32/08_finetune_280K@pair_T,0.15,G=0.025:2023-05-31-13-22-36-278997",
    "logs/cifar10_32/08_finetune_280K@pair_T,0.20,G=0.025:2023-05-31-13-22-49-172142",
    
    # "logs/celeba_64/00_baseline@uniform,G=0.0:2023-04-08-21-25-49-813419",
    # "logs/celeba_64/06_k=0.1@G=0.01:2023-04-15-18-47-28-701755",
    # "logs/celeba_64/07_finetune_300K@pair_T,0.10,G=0.025:2023-05-01-20-38-19-386865",
    # "logs/celeba_64/08_finetune_300K@pair_T,0.15,G=0.025:2023-05-30-01-30-20-698653",
    # "logs/celeba_64/08_finetune_300K@pair_T,0.20,G=0.025:2023-05-30-01-31-38-487658",
]
models = [
    # "ema_0.9999_250000.pt",
    # "ema_0.9999_260000.pt",
    # "ema_0.9999_270000.pt",
    # "ema_0.9999_280000.pt",
    # "ema_0.9999_290000.pt",

    # "ema_0.9999_005000.pt",
    # "ema_0.9999_010000.pt",
    "ema_0.9999_015000.pt",
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

