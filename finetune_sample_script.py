N = 4
B = 512
sampler = "ddim100,ddim50,ddim20,ddim10"
# sampler = "100,50,20,10"
project_dirs = [
    # "logs/celeba_64/00_baseline@uniform,G=0.0:2023-04-08-21-25-49-813419",
    # "logs/celeba_64/06_k=0.1@G=0.01:2023-04-15-18-47-28-701755",
    # "logs/celeba_64/07_finetune_200K@pair_T,0.15,G=0.025:2023-05-04-05-53-44-463583",
    # "logs/celeba_64/07_finetune_200K@pair_T,0.20,G=0.025:2023-05-05-01-29-18-655279",

    # "logs/cifar10_32/00_baseline@G=0.0:2023-04-28-22-46-37-350895",
    # "logs/cifar10_32/06_k=0.1@G=0.01:2023-04-29-05-56-57-541934",
    # "logs/cifar10_32/07_finetune_200K@pair_T,0.15,G=0.025:2023-05-02-07-41-59-080179"
    # "logs/cifar10_32/07_finetune_200K@pair_T,0.20,G=0.025:2023-05-02-00-05-29-687760",

    # "logs/celeba_64/07_finetune_200K@pair_T,0.15,G=0.025:2023-05-04-05-53-44-463583",
    # "logs/cifar10_32/07_finetune_200K@pair_T,0.15,G=0.025:2023-05-02-07-41-59-080179",
]
models = [
    # "ema_0.9999_200000.pt",
    # "ema_0.9999_300000.pt",
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
                 f"--model_path {project_dir} --pt_name {model} --num_samples 50000 --batch_size {B} --sampler {sampler}\n"
        # script = f"python -m torch.distributed.launch --nproc_per_node={N} scripts/image_pndm.py " + \
        #          f"--model_path {project_dir} --pt_name {model} --num_samples 50000 --batch_size {B} --sampler {sampler}\n"
        f.write(script)
f.close()

