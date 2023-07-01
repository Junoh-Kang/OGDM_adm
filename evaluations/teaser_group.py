from PIL import Image
import torchvision
import numpy as np
import torch as th
from glob import glob
import argparse 
import os
import tqdm
import random

folder = "logs/qual/teaser/individual"
def to_grid(image_num):
    tags_desc = ["base_ddim100", "base_ddim50", "base_ddim20", "base_ddim10",
                 "ours_ddim100", "ours_ddim50", "ours_ddim20", "ours_ddim10",
                 "ours_S-PNDM99", "ours_S-PNDM49", "ours_S-PNDM19", "ours_S-PNDM9"]

    tags_asc = ["base_ddim10", "base_ddim20", "base_ddim50", "base_ddim100",
                "ours_ddim10", "ours_ddim20", "ours_ddim50", "ours_ddim100",
                "ours_S-PNDM9", "ours_S-PNDM19", "ours_S-PNDM49", "ours_S-PNDM99"]
    
    images = []
    for tag in tags_desc:
        images.append(np.array(Image.open(f"{folder}/{image_num}_{tag}.png")))
    images = np.stack(images)
    images = th.tensor(images.transpose(0,3,1,2))

    grid = torchvision.utils.make_grid(images, 4, padding=0).permute(1,2,0).numpy()
    Image.fromarray(grid).save(f"logs/qual/teaser/grid/{image_num}_desc.png")

    images = []
    for tag in tags_asc:
        images.append(np.array(Image.open(f"{folder}/{image_num}_{tag}.png")))
    images = np.stack(images)
    images = th.tensor(images.transpose(0,3,1,2))

    grid = torchvision.utils.make_grid(images, 4, padding=0).permute(1,2,0).numpy()
    Image.fromarray(grid).save(f"logs/qual/teaser/grid/{image_num}_asc.png")

for image_num in range(10,26):
    to_grid(image_num)