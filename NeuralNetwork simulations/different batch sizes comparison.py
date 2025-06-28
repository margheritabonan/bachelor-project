# -*- coding: utf-8 -*-
import argparse
import numpy as np
from pprint import pprint
from joblib import Parallel, delayed
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

# (from utils in the original repository)
def label_to_onehot(target, num_classes=100):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target

def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))



# settings of the experiment
torch.manual_seed(200)
experiments_number = 5 # number of experiments for each batch size
number_batch_sizes = 6 # batch sizes are going to be 2**i for i in range(number_batch_sizes)
n_iter = 50



def run_experiment(index_batch_size):

    batch_size = 2**index_batch_size 
    

    indices_images = [i for i in range(3, batch_size + 3)]  # indices of images to leak

    # argument parser
    parser = argparse.ArgumentParser(description='Deep Leakage from Gradients.')
    parser.add_argument('--indices', type=int, nargs='+', default=indices_images,  
                        help='The indices for leaking images on MNIST.')
    args = parser.parse_args()

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"


    #  MNIST dataset
    dst = datasets.MNIST("~/.torch", download=True)
    channels = 1

    # CIFAR100 dataset
    #dst = datasets.CIFAR100("~/.torch", download=True)
    #channels = 3
    tp = transforms.ToTensor()
    tt = transforms.ToPILImage()

    # load ground truth data for n images
    img_indices = args.indices
    n = len(img_indices)  # # of images
    gt_data = torch.stack([tp(dst[i][0]) for i in img_indices]).to(device)  
    gt_labels = torch.Tensor([dst[i][1] for i in img_indices]).long().to(device) 
    gt_onehot_labels = label_to_onehot(gt_labels)  

    # print class labels
    #print("Class labels are:", [dst.classes[dst[i][1]] for i in img_indices])

    net = LeNet(in_channels=channels).to(device)
    #print(net)

    restart_interval = 20
    

    net.apply(weights_init)
    criterion = nn.CrossEntropyLoss().to(device) # the one used in iDLG

    #  original gradients
    pred = net(gt_data)
    y = criterion(pred, gt_onehot_labels)
    dy_dx = torch.autograd.grad(y, net.parameters())

    original_dy_dx = list((_.detach().clone() for _ in dy_dx))

    method = "DLG" # "DLG" or "iDLG"
    # note: iDLG can not be used for more than 1 image, so we use DLG for multiple images

    losses_experiments = []


    for experiment in range(experiments_number):  

        # dummy data and label
        dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)  # shape [n, 1, 28, 28]
        dummy_labels = torch.randn(gt_onehot_labels.size()).to(device).requires_grad_(True)  # shape is [n, 10]


        if method == 'DLG':
            optimizer = torch.optim.LBFGS([dummy_data, dummy_labels])
        elif method == 'iDLG':
            optimizer = torch.optim.LBFGS([dummy_data, ])
            # predict the ground-truth label
            label_pred = torch.argmin(torch.sum(original_dy_dx[-2], dim=-1), dim=-1).detach().reshape((n,)).requires_grad_(False)
            print("Predicted label is:", label_pred.item())

        history = {"initial": dummy_data.clone().detach().cpu(), "final": None}
        current_loss = 0
        
        loss_array = np.zeros(n_iter)
        for iters in range(n_iter):

            if current_loss >= 5 and (iters % restart_interval == 0): # or ((loss_array[iters - 10] - current_loss) < 1e-16 and iters > 10 and iters % restart_interval == 0):
                # re-initialize dummy data and labels
                #print("Restarting...")
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

            #print(iters, "%.10f" % current_loss)

            if iters == n_iter-1:  # save final reconstructed images
                history["final"] = dummy_data.clone().detach().cpu()
            
        losses_experiments.append(current_loss)
        # pixel wise loss:
        pixel_loss = torch.mean((dummy_data - gt_data) ** 2).item()
      

        print(f"Final pixel loss for batch size {batch_size}: {pixel_loss:.5e}, experiment: {experiment}")
    

    print(f"Standard deviation of losses for batch size {batch_size}: {np.std(losses_experiments, ddof=1)}")
   
    
    return [index_batch_size, np.mean(losses_experiments),  np.std(losses_experiments, ddof=1), np.min(losses_experiments), np.max(losses_experiments)]  

result = Parallel(n_jobs=12)(delayed(run_experiment)(index_batch_size) for index_batch_size in range(number_batch_sizes))

# order result by batch size
result.sort(key=lambda x: x[0])  # sort by index_batch_size

final_losses = np.array([res[1] for res in result])  
standard_deviation_experiments = np.array([res[2] for res in result])
min_losses = np.array([res[3] for res in result])
max_losses = np.array([res[4] for res in result])
 
err_lower = final_losses - min_losses
err_upper = max_losses - final_losses
asymmetric_error = np.array([err_lower, err_upper])

# plotting the results

batch_sizes = [2**i for i in range(number_batch_sizes)] 

plt.figure(figsize=(10, 10))

plt.plot(batch_sizes, final_losses, 'o-', label='Final Loss')
#plt.errorbar(batch_sizes, final_losses, yerr=asymmetric_error, fmt ="none", label= 'Range of values from experiments', capsize=4)

plt.legend() 
plt.title(f'Different batch sizes comparison: iterations = {n_iter}, experiments = {experiments_number}, dataset: MNIST')
plt.xlabel('Batch Size')
plt.ylabel('Loss function value')
plt.grid()
plt.show()


plt.figure(figsize=(10, 5))
plt.plot(batch_sizes, final_losses, 'o-', label='Final Loss')
#plt.errorbar(batch_sizes, final_losses, yerr=asymmetric_error, fmt ="none", label= 'Range of values from experiments', capsize=4)

#set log scale base 2
plt.xscale('log', base=2)
# plt.errorbar(batch_sizes, final_losses, yerr=standard_deviation_experiments, fmt='o', color='blue', capsize=5)

#plt.plot(batch_sizes, final_pixel_losses, 'o-', label='Final Pixel Loss')
#plt.errorbar(batch_sizes,final_pixel_losses,yerr=standard_deviation_pixel_experiments,fmt='o',color='red',capsize=5)
plt.legend() 
plt.title(f'Different batch sizes comparison (log-log plot): iterations = {n_iter}, experiments = {experiments_number}, dataset: CIFAR100')
plt.xlabel('Batch Size')
plt.ylabel('Loss function value')
plt.grid()
plt.show()

        



