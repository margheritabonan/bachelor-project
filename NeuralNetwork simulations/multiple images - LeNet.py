# -*- coding: utf-8 -*-
import argparse
import numpy as np
from pprint import pprint

from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import models, datasets, transforms
# init the model
from models.vision import LeNet, weights_init

# code adapted from https://github.com/mit-han-lab/dlg

# (from utils)
def label_to_onehot(target, num_classes=100):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target

def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))


print(torch.__version__, torchvision.__version__)


# argument parser
parser = argparse.ArgumentParser(description='Deep Leakage from Gradients.')
parser.add_argument('--indices', type=int, nargs='+', default=[50],  # insert the indeces of the images you want
                    help='The indices for leaking images on MNIST.')
args = parser.parse_args()

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
print("Running on %s" % device)

#  MNIST dataset
#dst = datasets.MNIST("~/.torch", download=True)
#channels = 1

# CIFAR100 dataset
dst = datasets.CIFAR100("~/.torch", download=True)
channels = 3
tp = transforms.ToTensor()
tt = transforms.ToPILImage()

# load ground truth data for n images
img_indices = args.indices
n = len(img_indices)  # # of images
gt_data = torch.stack([tp(dst[i][0]) for i in img_indices]).to(device)  
gt_labels = torch.Tensor([dst[i][1] for i in img_indices]).long().to(device) 
gt_onehot_labels = label_to_onehot(gt_labels)  

# print class labels
print("Class labels are:", [dst.classes[dst[i][1]] for i in img_indices])


net = LeNet(in_channels=channels).to(device)
print(net)

restart_interval = 20
torch.manual_seed(123)

net.apply(weights_init)
criterion = nn.CrossEntropyLoss().to(device) # the one used in iDLG
#criterion = cross_entropy_for_onehot

#  original gradients
pred = net(gt_data)
y = criterion(pred, gt_onehot_labels)
dy_dx = torch.autograd.grad(y, net.parameters())

original_dy_dx = list((_.detach().clone() for _ in dy_dx))

method = "DLG" # "DLG" or "iDLG"
# note: iDLG can not be used for more than 1 image


history_complete = [] 
history_labels = []

# dummy data and label
dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)  # shape [n, 1, 28, 28]
dummy_labels = torch.randn(gt_onehot_labels.size()).to(device).requires_grad_(True)  # shape is [n, 10]
history_labels.append(dummy_labels.clone().detach().cpu())

history = {"initial": dummy_data.clone().detach().cpu(), "final": None}

if method == 'DLG':
    optimizer = torch.optim.LBFGS([dummy_data, dummy_labels])
elif method == 'iDLG':
    optimizer = torch.optim.LBFGS([dummy_data, ])
    # predict the ground-truth label
    label_pred = torch.argmin(torch.sum(original_dy_dx[-2], dim=-1), dim=-1).detach().reshape((n,)).requires_grad_(False)
    print("Predicted label is:", label_pred.item())



current_loss = 0
n_iter = 300
loss_array = np.zeros(n_iter)
for iters in range(n_iter):

    if current_loss >= 5 and (iters % restart_interval == 0): # or ((loss_array[iters - 10] - current_loss) < 1e-16 and iters > 10 and iters % restart_interval == 0):
        # re-initialize dummy data and labels
        print("Restarting...")
        dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
        dummy_labels = torch.randn(gt_onehot_labels.size()).to(device).requires_grad_(True)
        if method == 'DLG':
            optimizer = torch.optim.LBFGS([dummy_data, dummy_labels])
        elif method == 'iDLG':
            optimizer = torch.optim.LBFGS([dummy_data, ])
            label_pred = torch.argmin(torch.sum(original_dy_dx[-2], dim=-1), dim=-1).detach().reshape((1,)).requires_grad_(False)


    def closure():
        optimizer.zero_grad()
        dummy_pred = net(dummy_data)

        if method == 'DLG':
            dummy_loss = criterion(dummy_pred, F.softmax(dummy_labels, dim=-1))

        elif method == 'iDLG':
            dummy_loss = criterion(dummy_pred, label_pred)

        dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)

        loss = 0
        for gx, gy in zip(dummy_dy_dx, original_dy_dx):
            loss += ((gx - gy) ** 2).sum()
        loss.backward()

        return loss

    optimizer.step(closure)
    current_loss = closure().item()
    loss_array[iters] = current_loss
    history_complete.append(dummy_data.detach().cpu().clone())
    history_labels.append(dummy_labels.detach().cpu().clone())

    print(iters, "%.10f" % current_loss)

    if iters == 299:  # save final reconstructed images
        history["final"] = dummy_data.clone().detach().cpu()

# plot the results
plt.figure(figsize=(8, 3 * n))  # adjust height based on the number of images
plt.title(f'{method} on {dst.__class__.__name__}, {n} images \n')
plt.subplots_adjust(hspace=1)
plt.axis('off')

for i in range(n):
    # original image
    plt.subplot(n, 3, i * 3 + 1)
    plt.imshow(tt(gt_data[i].cpu()))
    if i == 0:
        plt.title(f"Original")
    plt.axis('off')

    # initial dummy data
    plt.subplot(n, 3, i * 3 + 2)
    plt.imshow(tt(history["initial"][i]))
    if i == 0:
        plt.title(f"Initial dummy image")
    plt.axis('off')

    # final reconstructed image
    plt.subplot(n, 3, i * 3 + 3)
    plt.imshow(tt(history["final"][i]))
    if i == 0:
        plt.title(f"Reconstructed image")
    plt.axis('off')

plt.tight_layout()
plt.show()

#history_complete = np.array(history_complete)

if n == 1:

    plt.figure(figsize=(10, 5))
    plt.title(f'{method} on {dst.__class__.__name__}')
    plt.axis('off')
    # plot iterations images 0, 10, 20, ..., 90
    for i in range(9):
        plt.subplot(2, 5, i+1)
        plt.imshow(tt(history_complete[i * 30][0]))
        plt.title(f"Iter {i * 30}")
        # below the image, write
        # if dataset is MNIST, write the label
        if dst.__class__.__name__ == 'MNIST':
            plt.text(x = 0.5 ,y = -0.1, s= f"label: {history_labels[i * 30][0].argmax().item()}", transform=plt.gca().transAxes, ha='center', fontsize=10)
        elif dst.__class__.__name__ == 'CIFAR100':
            label_idx = history_labels[i * 30][0].argmax().item()
            label_str = dst.classes[label_idx]
            plt.text(x = 0.5 ,y = -0.1, s= f"label: {label_str}", transform=plt.gca().transAxes, ha='center', fontsize=10)
        plt.axis('off')
    plt.subplot(2, 5, 10)
    plt.imshow(tt(gt_data[0].cpu()))
    plt.title("Ground Truth")
    plt.axis('off')
    plt.tight_layout()
    plt.show()



