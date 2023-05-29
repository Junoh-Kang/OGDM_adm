import time    
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-n', type=int)
args = parser.parse_args()
for i in range(int(args.n * 3600)):
    time.sleep(1)