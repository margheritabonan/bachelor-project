import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import argparse
import math
import numpy as np

# code adapted from https://github.com/mit-han-lab/dlg

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on", device)

parser = argparse.ArgumentParser()
parser.add_argument('--index', type=int, default=50, help='Starting index for the 5 images from CIFAR-100')
args = parser.parse_args()


dst = datasets.CIFAR100("~/.torch", download=True)
tp = transforms.ToTensor()
tt = transforms.ToPILImage()


number_of_images = 2
img_indices = [args.index, args.index + 50] #, args.index + 2, args.index + 3, args.index + 4]
gt_data = torch.stack([tp(dst[i][0]) for i in img_indices]).to(device)  # Shape: [5, 3, 32, 32]
gt_target = torch.randn(number_of_images).to(device)  # Random target values for each image

# lin regression model for multiple images
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)  # one bias for all images

    def forward(self, x):
        return self.linear(x.view(x.size(0), -1))  # flatten each image in the batch

#  setup
model = LinearRegressionModel(input_dim=3 * 32 * 32, output_dim=1).to(device)
torch.manual_seed(12456)
model.apply(lambda m: nn.init.normal_(m.weight) if isinstance(m, nn.Linear) else None)
criterion = nn.MSELoss()

# restart frequency
restart_interval = 100

# initialize dummy input (number_of_images images)
#dummy_data = torch.randn_like(gt_data, requires_grad=True).to(device)
dummy_data = (gt_data + 0.1 * torch.randn_like(gt_data)).to(device).requires_grad_(True)

pred = model(gt_data)  # Forward pass
loss = criterion(pred.squeeze(), gt_target)  
dy_dx = torch.autograd.grad(loss, model.parameters())
original_grads = [g.detach().clone() for g in dy_dx]
dummy_target = torch.randn_like(gt_target).to(device).requires_grad_(True)  # Dummy target value (y) 
# optimizer
optimizer = torch.optim.LBFGS([dummy_data, dummy_target])
history = []


loss_list = []

for iters in range(3000):

    def closure():
        optimizer.zero_grad()

        pred = model(dummy_data)
        loss = criterion(pred.squeeze(), gt_target)  # Match target shape

        #  gradients for dummy data
        dummy_grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)

        # gradient matching loss
        grad_loss_mse = sum(((gx - gy) ** 2).mean() for gx, gy in zip(dummy_grads, original_grads))
        grad_loss = grad_loss_mse       


        # total loss
        total_loss = grad_loss  # if wanted, add regularization terms here
        total_loss.backward()
        return total_loss

    optimizer.step(closure)
    current_loss = closure().item()
    print(f"Iter {iters}: loss = {current_loss:.11f}")
    loss_list.append(current_loss)

    # restart if stagnating
    if (current_loss >= 1 and (iters % restart_interval == 0)) or math.isnan(current_loss) or (iters % 300 == 0):
        print("Restarting...")
        # Re-initialize dummy data
        dummy_data.data = torch.randn_like(gt_data).to(device)
        dummy_data = (gt_data + 0.1 * torch.randn_like(gt_data)).to(device).requires_grad_(True)
        optimizer = torch.optim.LBFGS([dummy_data, dummy_target])
    
    if iters %  100 == 0:
        history.append(dummy_data.detach().cpu().clone())

    # break loop if loss is small
    if current_loss < 1e-6:
        print("Converged")
        break

    pixel_loss = F.mse_loss(dummy_data, gt_data).item()

    if pixel_loss < 0.0001:
        print("Converged based on pixel similarity")
        break

# turn the loss list to array
loss_list = np.array(loss_list)
 
print("Reconstructed target values:")
for i in range(number_of_images):
    print(dummy_target[i].detach().cpu().item())

print("Original target values:")
for i in range(number_of_images):
    print(gt_target[i].detach().cpu().item())

plt.figure(figsize=(12, 6))
plt.title("Original Images", fontsize=16)
plt.axis('off')
plt.subplot(1, 2, 1)
for i in range(number_of_images):
    plt.subplot(1, number_of_images, i + 1)
    plt.axis('off')
    plt.imshow(tt(gt_data[i].cpu()))
    plt.axis('off')
plt.show()

plt.figure(figsize=(12, 6))
plt.title(f"Reconstructed images, loss: {loss_list[-1]:.5e}. ", fontsize=16)
plt.axis('off')
plt.subplot(1, 2, 1)
for i in range(number_of_images):
    plt.subplot(1, number_of_images, i + 1)
    plt.imshow(tt(history[-1][i]))
    plt.axis('off')
plt.show()

plt.figure(figsize=(12, 6))
plt.title(f"Dummy initialization, loss: {loss_list[0]:.5e}", fontsize=16)
plt.axis('off')
plt.subplot(1, 2, 1)
for i in range(number_of_images):
    plt.subplot(1, number_of_images, i + 1)
    plt.imshow(tt(dummy_data[i].cpu()))
    plt.axis('off')
plt.show()



