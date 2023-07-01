N = 1
B = 16
eta=0.0

sampler = "ddim100,ddim50,ddim20,ddim10,S-PNDM99,S-PNDM49,S-PNDM19,S-PNDM9"
# sampler = "ddimq50,ddimq20,ddimq10,S-PNDM49,S-PNDM19,F-PNDM41,F-PNDM11"
# sampler = "S-PNDM49,S-PNDM19,F-PNDM41,F-PNDM11"
# sampler="S-PNDM9"
project_dirs = [
    # "logs/cifar10_32/baseline",
    # "logs/cifar10_32/ours_scratch_0.10,0.01",
    # "logs/cifar10_32/ours_ft280k_0.10,0.025",
    # "logs/cifar10_32/ours_ft280k_0.15,0.025",
    # "logs/cifar10_32/ours_ft280k_0.20,0.025",
    

    "logs/celeba_64/baseline",
    "logs/celeba_64/ours_scratch_0.10,0.01",
    # "logs/celeba_64/ours_ft300k_0.10,0.025",
    # "logs/celeba_64/ours_ft300k_0.15,0.025",
    # "logs/celeba_64/ours_ft300k_0.20,0.025",

    # "logs/lsun/lsun_bedroom",
    # "logs/lsun/lsun_cat",
    # "logs/lsun/lsun_horse",
]
models = [
    # "ema_0.9999_210000.pt",
    # "ema_0.9999_280000.pt",
    "ema_0.9999_300000.pt",
    # "ema_0.9999_015000.pt",

    # "ema_0.9999_250000.pt"

    # "ema.pt"
]

f = open("./evaluations/script.txt", 'w')
for project_dir in project_dirs:
    for model in models:   
        script = f"python -m torch.distributed.launch --nproc_per_node={N} scripts/image_sample.py " + \
                 f"--project_dir {project_dir} --pt_name {model} --num_samples 50000 --batch_size {B} --sampler {sampler} --eta {eta}\n"
        f.write(script)
f.close()

