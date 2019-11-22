import argparse
import os

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

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST pruning from deep compression paper')
parser.add_argument('--batch-size', type=int, default=8, metavar='N',
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
parser.add_argument('--sensitivity', type=float, default=1,
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
model = Net(mask=True).to(device)
model.load_state_dict(torch.load('/content/gdrive/My Drive/data/model_L1_110e.ckpt', map_location="cpu")) # 경록

print(model)
# util.print_model_parameters(model)

# NOTE : `weight_decay` term denotes L2 regularization loss term
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
# optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001) # 경록
initial_optimizer_state_dict = optimizer.state_dict()

def train(epochs):
    model.train()
    for epoch in range(epochs):
        pbar = tqdm(enumerate(dl), total=len(dl))
        for batch_idx, (data, target) in pbar:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = model_utils.depth_loss(output, target)
            loss.backward()

            # zero-out all the gradients corresponding to the pruned connections
            for name, p in model.named_parameters():
                if 'mask' in name:
                    continue
                tensor = p.data.cpu().numpy()
                grad_tensor = p.grad.data.cpu().numpy()
                grad_tensor = np.where(tensor==0, 0, grad_tensor)
                p.grad.data = torch.from_numpy(grad_tensor).to(device)

            optimizer.step()
            if batch_idx % args.log_interval == 0:
                done = batch_idx * len(data)
                percentage = 100. * batch_idx / len(dl)
                pbar.set_description(f'Train Epoch: {epoch} [{done:5}/{len(dl.dataset)} ({percentage:3.0f}%)]  Loss: {loss.item():.6f}')


def test():
    model.eval()
    test_loss = 0
    with torch.no_grad():
        print(f'tl length:{len(tl.dataset):.4f}')
        for data, target in tl:
            data, target = data.to(device), target.to(device)
            output = model(data)

            test_loss += model_utils.depth_loss(output, target).item()

        test_loss /= len(tl.dataset)
        print('test is over')
        print(f'Test set: Average loss: {test_loss:.4f}')
    return test_loss


print("--- Before pruning ---")
util.print_nonzeros(model)

# Pruning
########################################################################################################################
############################################################
############################################################여기서 perentile할 건지, std할건지 정해야 한다.
########################################################################################################################
# model.prune_by_percentile(args.percentile) #경록

i=2.0
for num in range(0, 8, 1):
    i += 0.2
    print(f"sensetivity : {i}")
    model.prune_by_std(i)
    print("--- After pruning ---")
    accuracy = test()
    util.log(args.log, f"accuracy_after_pruning {accuracy}")
    util.print_nonzeros(model)



# percentile
# i=80
# for num in range(0, 20, 1):
#     i += 1
#     print(f"percentile : {i}")
#     model.prune_by_percentile(i)
#     print("--- After pruning ---")
#     accuracy = test()
#     util.log(args.log, f"accuracy_after_pruning {accuracy}")
#     util.print_nonzeros(model)


# # Retrain
# print("---0th  Retraining ---")
# optimizer.load_state_dict(initial_optimizer_state_dict) # Reset the optimizer
# train(args.epochs)
# accuracy = test()
# util.log(args.log, f"accuracy_after_retraining {accuracy}")
# torch.save(model, f"/content/gdrive/My Drive/data/model_L1_150e_pr_std1_rt_1.ptmodel") # 경록
# torch.save(model.state_dict(), '/content/gdrive/My Drive/data/model_L1_150e_pr_std1_rt_1.ckpt')# 경록
#
# print("--- 0th After Retraining ---")
# util.print_nonzeros(model)
