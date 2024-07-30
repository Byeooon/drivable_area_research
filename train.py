import torch
import random
import numpy as np
from drivable_learner import DrivableLearner
import os
import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, required=False, default='/home/julio981007/HDD/orfd')
parser.add_argument("--ckpt_dir", type=str, default="./checkpoints/tmp")
parser.add_argument("--patience", type=int, default=10)
parser.add_argument("--learning_rate", type=float, default=1e-4, help='Learning rate of for adam')
parser.add_argument("--batch_size", type=int, default=6)
parser.add_argument("--num_epochs", type=int, default=1000)

parser.add_argument("--img_height", type=int, default=644)
parser.add_argument("--img_width", type=int, default=644)
parser.add_argument("--depth", type=str2bool, default=False)

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

    os.environ["XFORMERS_DISABLED"] = "1" # Switch to enable xFormers
    USE_CUDA = torch.cuda.is_available()
    print(USE_CUDA)
    device = torch.device('cuda:0' if USE_CUDA else 'cpu')
    if device=="cuda": torch.cuda.empty_cache()
    print('학습을 진행하는 기기:',device)
    
    learner = DrivableLearner(args, device)
    learner.train()