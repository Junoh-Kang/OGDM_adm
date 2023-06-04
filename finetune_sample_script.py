N = 4
B = 512
eta=0.0

sampler = "ddim10"
# sampler = "ddimq50,ddimq20,ddimq10"
# sampler = "41,11"
project_dirs = [
    # "logs/cifar10_32/baseline_280k"
    "logs/cifar10_32/ours_scratch_0.10,0.01_210k",
]
models = [
    "ema_0.9999_210000.pt",
    # "ema_0.9999_280000.pt",
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

