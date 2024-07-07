import torch
import random
import numpy as np
from drivable_learner import DrivableLearner
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, required=False)
parser.add_argument("--ckpt_dir", type=str, default="./checkpoints/")
parser.add_argument("--patience", type=int, default=10)
parser.add_argument("--learning_rate", type=float, default=1e-4, help='Learning rate of for adam')
parser.add_argument("--batch_size", type=int, default=6)
parser.add_argument("--num_epochs", type=int, default=1)

parser.add_argument("--img_height", type=int, default=128)
parser.add_argument("--img_width", type=int, default=416)

parser.add_argument("--smooth_weight", type=float, default=0.1, help='Weight for smoothness')
parser.add_argument("--seq_length", type=int, default=3)
parser.add_argument("--num_source", type=int, default=2)
parser.add_argument("--num_scales", type=int, default=4)
parser.add_argument("--save_latest_freq", type=int, default=5000)
parser.add_argument("--summary_freq", type=int, default=100)
parser.add_argument("--max_to_keep", type=int, default=5)

args = parser.parse_args()


if __name__ == '__main__':
    seed = 34
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    print(args)

    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    learner = DrivableLearner(args)
    learner.train()