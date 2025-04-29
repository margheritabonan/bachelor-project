import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from PIL import Image
import argparse
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['axes.titlesize'] = 8# set global font size for subplot titles

# code adapted from https://github.com/mit-han-lab/dlg

parser = argparse.ArgumentParser()
parser.add_argument('--index', type=int, default=100, help='Image index from CIFAR') # 50 shark. 100 other animal
parser.add_argument('--image', type=str, default="", help='Optional custom image path')
args = parser.parse_args()


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on", device)

# images
dst = datasets.CIFAR100("~/.torch", download=True)
tp = transforms.ToTensor()
tt = transforms.ToPILImage()

img_index = args.index
gt_data = tp(dst[img_index][0]).to(device).unsqueeze(0)  # shape: [1, 3, 32, 32]
gt_target = torch.randn(1, 1).to(device)  #  random continuous target value

if args.image:
    gt_data = Image.open(args.image)
    gt_data = tp(gt_data).to(device).unsqueeze(0)


# inspired by https://medium.com/data-science/linear-regression-with-pytorch-eb6dedead817 

# linear regression model: this makes absolutely no sense as a model as we take a random y
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim = 1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)  # outpus is a random scalar

    def forward(self, x):
        return self.linear(x.view(x.size(0), -1))  

# model setup
model = LinearRegressionModel(input_dim=3*32*32).to(device)
torch.manual_seed(1245)
model.apply(lambda m: nn.init.normal_(m.weight) if isinstance(m, nn.Linear) else None)
criterion = nn.MSELoss()

#  original gradient
pred = model(gt_data)
loss = criterion(pred, gt_target)
dy_dx = torch.autograd.grad(loss, model.parameters())

#original_grads = [g.detach().clone() for g in dy_dx]

#  weights (w) and bias (b) from the linear layer
w = model.linear.weight.detach().clone().cpu().numpy().flatten()  
b = model.linear.bias.detach().clone().item()    
y = pred.detach().clone().item()

# gradient of the loss with respect to the weights and bias
grad_w = dy_dx[0].cpu().numpy().flatten() # Gradient of loss w.r.t. weights (same shape as w)

#print("Weights (w):", w)
#print("Bias (b):", b)
#print("Gradient of loss w.r.t. weights (grad_w):", grad_w)
#print("y:" , y)

# calculate the scaling alpha for which gradient = alpha * data

alpha = 1* ((b - y) + np.sqrt((b - y)**2 + 4 * np.dot(w.T, grad_w))) # double the one in the calculations
print("alpha calc:", alpha)

alpha_inv = 1/alpha

# get the original data as g/alpha
retrieved_data = grad_w * alpha_inv

# reshape oringal data to image
retrieved_data = retrieved_data.reshape(3, 32, 32)

#normalize
#retrieved_data = (retrieved_data - retrieved_data.min()) / (retrieved_data.max() - retrieved_data.min())
print("min:", retrieved_data.min())
print("max:", retrieved_data.max())
# plot original image and retrieved image 
plt.subplot(2, 3, 1)
plt.axis('off')
plt.title("Original")
plt.imshow(tt(gt_data[0].cpu()))

plt.subplot(2, 3, 2)
plt.axis('off')
plt.title(r"$\alpha$-retrieved")
plt.imshow(retrieved_data.transpose(1, 2, 0))

plt.subplot(2, 3, 3)
plt.axis('off')
grad_w = grad_w.reshape(3, 32, 32)
# normalize the gradient
grad_w_norm = (grad_w - grad_w.min()) / (grad_w.max() - grad_w.min())
plt.title("Normalized gradient")
plt.imshow(grad_w_norm.transpose(1, 2, 0))


plt.subplot(2, 3, 4)
plt.axis('off')
plt.title("Gradient non normalized")
plt.imshow(grad_w.transpose(1, 2, 0))


plt.subplot(2, 3, 5)
plt.axis('off')
plt.title(r"100 x ($\alpha$-retrieved - original)")
diff = 100*(retrieved_data - gt_data[0].cpu().numpy())
plt.imshow(diff.transpose(1, 2, 0))

plt.subplot(2, 3, 6)
plt.axis('off')
plt.title("100 x (normalized grad - orig)")
diff2 = 100*( grad_w_norm - gt_data[0].cpu().numpy())
plt.imshow(diff2.transpose(1, 2, 0))


plt.tight_layout()

plt.show()







