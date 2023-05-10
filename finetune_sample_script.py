N = 2
B = 512
sampler = "ddim100,ddim50"
# sampler = "100,50,20,10"
project_dirs = [
    "logs/cifar10_32/07_finetune_200K@pair_T,0.10,G=0.025:2023-05-02-15-17-01-350808",
    "logs/celeba_64/07_finetune_200K@pair_T,0.10,G=0.025:2023-05-03-11-13-16-877481",
]
models = [
    # "ema_0.9999_200000.pt"
    # "ema_0.9999_005000.pt",
    # "ema_0.9999_010000.pt",
    # "ema_0.9999_015000.pt",
    # "ema_0.9999_020000.pt",
    # "ema_0.9999_025000.pt",
    "ema_0.9999_030000.pt",
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

