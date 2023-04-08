echo "some data for the file" >> "test.txt"

CUDA_VISIBLE_DEVICES=1 python -m pytorch_fid ./data/celeba_64_ref.npz logs/celeba_64/05_sampler@pair_t,0.10,G=0.01:2023-03-30-23-27-29-366818/fid/ema_0.9999_300000.pt/ddim10 --batch-size 512
CUDA_VISIBLE_DEVICES=1 python -m pytorch_fid ./data/celeba_64_ref.npz logs/celeba_64/05_sampler@pair_T,0.10,G=0.01:2023-03-31-13-56-51-323496/fid/ema_0.9999_300000.pt/ddim10 --batch-size 512
CUDA_VISIBLE_DEVICES=1 python -m pytorch_fid ./data/celeba_64_ref.npz logs/celeba_64/05_sampler@pair_T,0.20,G=0.005:2023-03-31-01-46-04-169445/fid/ema_0.9999_300000.pt/ddim10 --batch-size 512
CUDA_VISIBLE_DEVICES=1 python -m pytorch_fid ./data/celeba_64_ref.npz logs/celeba_64/05_sampler@pair_t,0.20,G=0.005:2023-03-31-02-13-33-236817/fid/ema_0.9999_300000.pt/ddim10 --batch-size 512





python -m pytorch_fid data/celeba_64_ref logs/celeba_64/05_sampler@pair_t,0.10,G=0.01:2023-03-30-23-27-29-366818/fid/ema_0.9999_300000.pt/ddim5
python -m pytorch_fid data/celeba_64_ref logs/celeba_64/05_sampler@pair_t,0.10,G=0.01:2023-03-30-23-27-29-366818/fid/ema_0.9999_300000.pt/ddim10
python -m pytorch_fid data/celeba_64 logs/celeba_64/05_sampler@pair_t,0.10,G=0.01:2023-03-30-23-27-29-366818/fid/ema_0.9999_300000.pt/ddim100
python -m pytorch_fid data/celeba_64 logs/celeba_64/05_sampler@pair_t,0.10,G=0.01:2023-03-30-23-27-29-366818/fid/ema_0.9999_300000.pt/F-PNDM5
python -m pytorch_fid data/celeba_64 logs/celeba_64/05_sampler@pair_t,0.10,G=0.01:2023-03-30-23-27-29-366818/fid/ema_0.9999_300000.pt/F-PNDM10
python -m pytorch_fid data/celeba_64 logs/celeba_64/05_sampler@pair_t,0.10,G=0.01:2023-03-30-23-27-29-366818/fid/ema_0.9999_300000.pt/F-PNDM100