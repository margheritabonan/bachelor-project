import argparse
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt

from torch.autograd.functional import hvp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import models, datasets, transforms
# init the model
from models.vision import LeNet

###############################################
torch.manual_seed(12346)
###########################################

# setting up model & data

# (from utils)
def label_to_onehot(target, num_classes=100): # num classes has to be adapted for the different models/datasets. here cifar has 10
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target

def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))


# argument parser
parser = argparse.ArgumentParser(description='Deep Leakage from Gradients.')
parser.add_argument('--indices', type=int, nargs='+', default=[7],  # insert the indeces of the images you want
                    help='The indices for images on MNIST/CIFAR100.')
args = parser.parse_args()

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
print("Running on %s" % device)

#  MNIST dataset
#dst = datasets.MNIST("~/.torch", download=True)
#channels = 1

# CIFAR100 dataset
dst = datasets.CIFAR100("~/.torch", download=False)
channels = 3

# network
net = LeNet(channels).to(device) 

# here ground truth data is loaded
tp = transforms.ToTensor()
tt = transforms.ToPILImage()

# load ground truth data for n images
img_indices = args.indices
n = len(img_indices)  # amound of images
gt_data = torch.stack([tp(dst[i][0]) for i in img_indices]).to(device)  
gt_labels = torch.tensor([dst[i][1] for i in img_indices], dtype=torch.long, device=device)

####################################################
# criterion for classification
criterion = nn.CrossEntropyLoss().to(device) # the one used in iDLG

##########################################################
# parameters operations needed for later computations.
def flat_params(params):
    return torch.cat([p.view(-1) for p in params])

def set_flat_params(params, flat_x):
    idx = 0
    for p in params:
        num = p.numel()
        p.data.copy_(flat_x[idx:idx+num].view_as(p))
        idx += num


##########################################################
#  "attack" loss ( difference between dummy and real gradients)

def attack_loss_wrt_dummy_data(dummy_data, net, dummy_labels, criterion, original_dy_dx, device):
    """
    attack loss = || grad_{net.params}(L(dummy_data)) - original_dy_dx ||^2
    we treat dummy_data as the variable and fix the network parameters and the labels.
    """
    
    out = net(dummy_data)
    loss = criterion(out, dummy_labels)
    
    grads = torch.autograd.grad(loss, net.parameters(), create_graph=True)
    g_dummy = torch.cat([g.view(-1) for g in grads])
    g_star = torch.cat([g.view(-1) for g in original_dy_dx]).to(device)

    return (g_dummy - g_star).pow(2).sum()

def hessian_vector_product_dummy_data(dummy_data, vec, net, dummy_labels, criterion, original_dy_dx, device):
    """
    computes H·v where H is the Hessian of the attack loss wrt dummy_data.
    """
    # We define a closure that returns attack_loss_dummy_data as a function of dummy_data only.
    def closure(x):
        # Reshape x to match dummy_data’s shape in case we flatten it
        x_reshaped = x.view_as(dummy_data).requires_grad_(True)
        return attack_loss_wrt_dummy_data(x_reshaped, net, dummy_labels, criterion, original_dy_dx, device)

    v_reshaped = vec.view_as(dummy_data)
    _, Hv = hvp(closure, dummy_data, v_reshaped) # for a function f, hpv(f,x,v) computes Hv where H is the hessian of f at x
    return Hv

##########################################################
# power method to get the minimum eigenvalue of Hessian.

def min_eigenvalue_dummy_data(dummy_data, net, dummy_labels, criterion, original_dy_dx, device,
                              power_iters=50, tol=1e-6):
    """
    power method to find the minimum eigenvalue of the Hessian wrt dummy_data, at an initial point.
    """
    # dd (dummy data) is the variable we optimize over
    dd = dummy_data.clone().detach().requires_grad_(True)
    D = dd.numel() # numel gives the number of elements in the tensor

    v = torch.randn(D, device=device)
    v /= v.norm()
    mu_old = None

    for i in range(power_iters):
        Hv = hessian_vector_product_dummy_data(dd, v, net, dummy_labels, criterion, original_dy_dx, device)
        w = -Hv  # we look at max eigenvalue of -H, which is min eigenvalue of H
        w_norm = w.norm(p=2).clamp_min(1e-12)
        v = w / w_norm

        Hv2 = hessian_vector_product_dummy_data(dd, v, net, dummy_labels, criterion, original_dy_dx, device)
    
        mu = v.view(-1).dot(Hv2.view(-1)).item() # this is the Rayleigh quotient (reshape to match the vector shape)
        if mu_old is not None and abs(mu - mu_old) < tol:
            print(f"converged at power iteration {i}")
            break
    
        mu_old = mu

    return mu

# compute the real gradients (original_dy_dx) from ground-truth
pred = net(gt_data)
y = criterion(pred, gt_labels)
dy_dx = torch.autograd.grad(y, net.parameters())
original_dy_dx = [_.detach().clone() for _ in dy_dx]

# initial point (to calculate the hesisan around)
#data_point= torch.randn((1, 1, 28, 28), device=device, requires_grad=True) # could do random sampling on some grid
#label = torch.randint(0, 10, size=(1,), device=device).long()

# use original picture and label as point
data_point = gt_data + torch.randn(gt_data.size()).to(device) * 0  #  perturbation
label = gt_labels
# actually compute the min eigenvalue of the attack loss Hessian


params = list(net.parameters())
min_eig = min_eigenvalue_dummy_data(
    data_point, # point around which we compute the Hessian
    net,
    label,
    criterion,
    original_dy_dx,
    device,
    power_iters=50,
    tol=1e-15
)

print(f"min eigenval of the attack (with fixed labels) Hessian: {min_eig}")

