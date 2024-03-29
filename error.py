import argparse
import os
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm

from net.models import Net
from net.quantization import apply_weight_sharing
import util

# 아래는 depth map prediction을 위해서 필용한 것.
import model_utils
from dataset import NYUDataset
from custom_transforms import *
import plot_utils

os.makedirs('saves', exist_ok=True)

parser = argparse.ArgumentParser(description='PyTorch MNIST pruning from deep compression paper')
parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 8)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 1)')
parser.add_argument('--lr', type=float, default=0.0000005, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--log', type=str, default='log.txt',
                    help='log file name')
parser.add_argument('--sensitivity', type=float, default=1.6,
                    help="sensitivity value that is multiplied to layer's std in order to get threshold value")
parser.add_argument('--percentile', type=float, default=85,
                    help="percentile value that is under % value to get threshold value")
args = parser.parse_args()
# Control Seed
torch.manual_seed(args.seed)

# Select Device
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else 'cpu')
if use_cuda:
    print("Using CUDA!")
    torch.cuda.manual_seed(args.seed)
else:
    print('Not using CUDA!!!')

# Loader
kwargs = {'num_workers': 5, 'pin_memory': True} if use_cuda else {}
# train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('data', train=True, download=True,
#                    transform=transforms.Compose([
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.1307,), (0.3081,))
#                    ])),
#     batch_size=args.batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('data', train=False, transform=transforms.Compose([
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.1307,), (0.3081,))
#                    ])),
#     batch_size=args.test_batch_size, shuffle=False, **kwargs)

# 아래는 depth Map Prediction을 위한
bs = 8
sz = (320,240)
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
mean, std = torch.tensor(mean), torch.tensor(std)
unnormalize = UnNormalizeImgBatch(mean, std)

tfms = transforms.Compose([
    ResizeImgAndDepth(sz),
    RandomHorizontalFlip(),
    ImgAndDepthToTensor(),
    NormalizeImg(mean, std)
])

ds = NYUDataset('/content/gdrive/My Drive/data/train.mat', tfms) # 경록
dl = torch.utils.data.DataLoader(ds, bs, shuffle=True)

# test loader 준비
ts = NYUDataset('/content/gdrive/My Drive/data/test.mat', tfms) # 경록
tl = torch.utils.data.DataLoader(ts, bs, shuffle=True)

# Define which model to use

#프룬되기 전
model = Net(mask=True).to(device)
model.load_state_dict(torch.load('/content/gdrive/My Drive/data/model_L2_190e.ckpt', map_location="cpu")) # 경록

#프룬된 후
model1 = Net(mask=True).to(device)
model1.load_state_dict(torch.load('/content/gdrive/My Drive/data/model_L2_190e_pr_re14.ckpt', map_location="cpu")) # 경록
criterion = nn.MSELoss()

# 록이꺼
# model_L2_190e_pr_re14
# 내꺼
# model_L1_110e_pr_re11

# print("--- Before pruning ---")
# util.print_nonzeros(model)
#
# print("--- After pruning ---")
# util.print_nonzeros(model1)

def test():
    model.eval()
    error_0 = 0 # scale-Invariant Error
    error_1 = 0 # RMS linear
    error_2 = 0 # RMS log
    error_3 = 0 # abs rel
    error_4 = 0 # sqr rel
    avg_psnr = 0  # psnr
    with torch.no_grad():
        for data, target in tl:
            data, target = data.to(device), target.to(device)
            output = model(data)
            error_0 += model_utils.depth_loss(output, target).item()
            target.squeeze_(dim=1)  # actual_depth 를
            error_1 += model_utils.err_rms_linear(output, target).item()
            target.squeeze_(dim=1)  # actual_depth 를
            error_2 += model_utils.err_rms_log(output, target).item()
            target.squeeze_(dim=1)  # actual_depth 를
            error_3 += model_utils.err_abs_rel(output, target).item()
            target.squeeze_(dim=1)  # actual_depth 를
            error_4 += model_utils.err_sql_rel(output, target).item()
            target.squeeze_(dim=1)  # actual_depth 를
            avg_psnr += model_utils.err_psnr(output, target).item()
            target.squeeze_(dim=1)  # actual_depth 를

        error_0 /= len(tl.dataset)
        error_1 /= len(tl.dataset)
        error_2 /= len(tl.dataset)
        error_3 /= len(tl.dataset)
        error_4 /= len(tl.dataset)
        avg_psnr /= len(tl.dataset)
        print(f'sclInvError:{error_0:.4f} RMSlinear:{error_1:.4f} RMSlog:{error_2:.4f} AbsRel:{error_3:.4f} SqrRel:{error_4:.4f} PSNR:{avg_psnr:.4f}')
    return error_1

def test1():
    model1.eval()
    error_0 = 0 # scale-Invariant Error
    error_1 = 0 # RMS linear
    error_2 = 0 # RMS log
    error_3 = 0 # abs rel
    error_4 = 0 # sqr rel
    avg_psnr = 0  # psnr
    with torch.no_grad():
        for data, target in tl:
            data, target = data.to(device), target.to(device)
            output = model1(data)
            error_0 += model_utils.depth_loss(output, target).item()
            target.squeeze_(dim=1)  # actual_depth 를
            error_1 += model_utils.err_rms_linear(output, target).item()
            target.squeeze_(dim=1)  # actual_depth 를
            error_2 += model_utils.err_rms_log(output, target).item()
            target.squeeze_(dim=1)  # actual_depth 를
            error_3 += model_utils.err_abs_rel(output, target).item()
            target.squeeze_(dim=1)  # actual_depth 를
            error_4 += model_utils.err_sql_rel(output, target).item()
            target.squeeze_(dim=1)  # actual_depth 를
            avg_psnr += model_utils.err_psnr(output, target).item()
            target.squeeze_(dim=1)  # actual_depth 를

        error_0 /= len(tl.dataset)
        error_1 /= len(tl.dataset)
        error_2 /= len(tl.dataset)
        error_3 /= len(tl.dataset)
        error_4 /= len(tl.dataset)
        avg_psnr /= len(tl.dataset)
        print(f'sclInvError:{error_0:.4f} RMSlinear:{error_1:.4f} RMSlog:{error_2:.4f} AbsRel:{error_3:.4f} SqrRel:{error_4:.4f} PSNR:{avg_psnr:.4f}')
    return error_1


accuracy = test()
accuracy1 = test1()


# def test():
#     model.eval()
#     error_0 = 0 # scale-Invariant Error
#     error_1 = 0 # RMS linear
#     error_2 = 0 # RMS log
#     error_3 = 0 # abs rel
#     error_4 = 0 # sqr rel
#     avg_psnr = 0  # psnr
#     with torch.no_grad():
#         data, target = next(iter(tl))
#         data, target = data.to(device), target.to(device)
#         output = model(data)
#         error_0 += model_utils.depth_loss(output, target).item()
#         target.squeeze_(dim=1)  # actual_depth 를
#         error_1 += model_utils.err_rms_linear(output, target).item()
#         target.squeeze_(dim=1)  # actual_depth 를
#         error_2 += model_utils.err_rms_log(output, target).item()
#         target.squeeze_(dim=1)  # actual_depth 를
#         error_3 += model_utils.err_abs_rel(output, target).item()
#         target.squeeze_(dim=1)  # actual_depth 를
#         error_4 += model_utils.err_sql_rel(output, target).item()
#         target.squeeze_(dim=1)  # actual_depth 를
#         avg_psnr += model_utils.err_psnr(output, target).item()
#         target.squeeze_(dim=1)  # actual_depth 를
#
#         error_0 /= 8
#         error_1 /= 8
#         error_2 /= 8
#         error_3 /= 8
#         error_4 /= 8
#         avg_psnr /= 8
#         print('test is over')
#         print(
#             f'scale-Invariant Error:{error_0:.4f} \nRMS linear : {error_1:.4f} \nRMS log : {error_2:.4f} \nabs rel : {error_3:.4f} \nsqr rel : {error_4:.4f}  \npsnr : {avg_psnr:.4f}')
#     return error_1
#
# def test1():
#     model1.eval()
#     error_0 = 0 # scale-Invariant Error
#     error_1 = 0 # RMS linear
#     error_2 = 0 # RMS log
#     error_3 = 0 # abs rel
#     error_4 = 0 # sqr rel
#     avg_psnr = 0  # psnr
#     with torch.no_grad():
#         data, target = next(iter(tl))
#         data, target = data.to(device), target.to(device)
#         output = model1(data)
#         error_0 += model_utils.depth_loss(output, target).item()
#         target.squeeze_(dim=1)  # actual_depth 를
#         error_1 += model_utils.err_rms_linear(output, target).item()
#         target.squeeze_(dim=1)  # actual_depth 를
#         error_2 += model_utils.err_rms_log(output, target).item()
#         target.squeeze_(dim=1)  # actual_depth 를
#         error_3 += model_utils.err_abs_rel(output, target).item()
#         target.squeeze_(dim=1)  # actual_depth 를
#         error_4 += model_utils.err_sql_rel(output, target).item()
#         target.squeeze_(dim=1)  # actual_depth 를
#         avg_psnr += model_utils.err_psnr(output, target).item()
#         target.squeeze_(dim=1)  # actual_depth 를
#
#         error_0 /= 8
#         error_1 /= 8
#         error_2 /= 8
#         error_3 /= 8
#         error_4 /= 8
#         avg_psnr /= 8
#         print('test is over')
#         print(f'scale-Invariant Error:{error_0:.4f} \nRMS linear : {error_1:.4f} \nRMS log : {error_2:.4f} \nabs rel : {error_3:.4f} \nsqr rel : {error_4:.4f}  \npsnr : {avg_psnr:.4f}')
#     return error_1
