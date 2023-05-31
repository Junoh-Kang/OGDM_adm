import argparse
from glob import glob
from PIL import Image
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('--dir1', type=str)
parser.add_argument('--dir2', type=str)
args = parser.parse_args()

files = glob(f"{args.dir1}/*.png")
images = []
for f in files:
    images.append(np.array(Image.open(f)))
try:
    arr = np.stack(images, axis=0)
    output_path = f"{args.dir2}"
    np.savez(output_path, arr[:50000])
    print("finished")
except:
    breakpoint()