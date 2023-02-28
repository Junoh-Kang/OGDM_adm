"""
Helpers for distributed training.
"""

import io
import os
import socket
import subprocess as sp

import blobfile as bf
from mpi4py import MPI
import torch as th
import torch.distributed as dist
import torch.multiprocessing as mp

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 4
SETUP_RETRY_COUNT = 3


def setup_dist(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    mp.set_start_method('spawn') 
    th.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, "env://"), flush=True)
    th.distributed.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, 
        world_size=args.world_size, rank=args.rank, )
        # timeout=datetime.timedelta(seconds=6000))
    th.distributed.barrier()


# def main(args):
# 		init_distributed_mode(args)
# 		seed = args.seed + torch.distributed.get_rank()
# 		torch.manual_seed(seed)
# 		np.random.seed(seed)



# def setup_dist():
#     """
#     Setup a distributed process group.
#     """
#     if dist.is_initialized():
#         return
#     os.environ["CUDA_VISIBLE_DEVICES"] = f"{MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE}"

#     comm = MPI.COMM_WORLD
#     backend = "gloo" if not th.cuda.is_available() else "nccl"

#     if backend == "gloo":
#         hostname = "localhost"
#     else:
#         hostname = socket.gethostbyname(socket.getfqdn())
    
#     os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
#     os.environ["RANK"] = str(comm.rank)
#     os.environ["WORLD_SIZE"] = str(comm.size)

#     port = comm.bcast(_find_free_port(), root=0)
#     os.environ["MASTER_PORT"] = str(port)
#     dist.init_process_group(backend=backend, init_method="env://")


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device(f"cuda")
    return th.device("cpu")

# FIXME(when load state dict later)
def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    chunk_size = 2 ** 30  # MPI has a relatively small size limit
    if dist.get_rank() == 0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
        num_chunks = len(data) // chunk_size
        if len(data) % chunk_size:
            num_chunks += 1
        # MPI.COMM_WORLD.bcast(num_chunks)
        dist.broadcast(num_chunks,0)
        for i in range(0, len(data), chunk_size):
            # MPI.COMM_WORLD.bcast(data[i : i + chunk_size])
            dist.broadcast(data[i : i + chunk_size], 0)
    else:
        # num_chunks = MPI.COMM_WORLD.bcast(None)
        num_chunks = dist.broadcast(None, 0)
        data = bytes()
        for _ in range(num_chunks):
            # data += MPI.COMM_WORLD.bcast(None)
            data += dist.broadcast(None, 0)
    return th.load(io.BytesIO(data), **kwargs)

# def load_state_dict(path, **kwargs):
#     """
#     Load a PyTorch file without redundant fetches across MPI ranks.
#     """
#     chunk_size = 2 ** 30  # MPI has a relatively small size limit
#     if dist.MPI.COMM_WORLD.Get_rank() == 0:
#         with bf.BlobFile(path, "rb") as f:
#             data = f.read()
#         num_chunks = len(data) // chunk_size
#         if len(data) % chunk_size:
#             num_chunks += 1
#         MPI.COMM_WORLD.bcast(num_chunks)
#         for i in range(0, len(data), chunk_size):
#             MPI.COMM_WORLD.bcast(data[i : i + chunk_size])
#     else:
#         num_chunks = MPI.COMM_WORLD.bcast(None)
#         data = bytes()
#         for _ in range(num_chunks):
#             data += MPI.COMM_WORLD.bcast(None)

#     return th.load(io.BytesIO(data), **kwargs)

def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)

def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
