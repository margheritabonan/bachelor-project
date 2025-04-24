import torch
import torch.nn as nn
from torchvision import datasets, transforms
from PIL import Image
import argparse
import matplotlib.pyplot as plt

# code adapted from https://github.com/mit-han-lab/dlg

parser = argparse.ArgumentParser()
parser.add_argument('--index', type=int, default=50, help='Image index from CIFAR')
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

# Model setup
model = LinearRegressionModel(input_dim=3*32*32).to(device)
torch.manual_seed(125)
model.apply(lambda m: nn.init.normal_(m.weight) if isinstance(m, nn.Linear) else None)
criterion = nn.MSELoss()

# Compute original gradient
pred = model(gt_data)
loss = criterion(pred, gt_target)
dy_dx = torch.autograd.grad(loss, model.parameters())

original_grads = [g.detach().clone() for g in dy_dx]

# Initialize dummy input
dummy_data = torch.randn_like(gt_data, requires_grad=True).to(device)

optimizer = torch.optim.LBFGS([dummy_data])
#optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

restart_interval = 10 # not used yet
history = []

for iters in range(60):

    def closure():

    # Clear gradient buffers because we don't want any gradient from previous epoch to 
    # carry forward, dont want to cummulate gradients
        optimizer.zero_grad()

        pred = model(dummy_data)
        loss = criterion(pred, gt_target)
        dummy_grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)

        grad_loss = sum(((gx - gy)**2).sum() for gx, gy in zip(dummy_grads, original_grads))
        grad_loss.backward()
        return grad_loss

    optimizer.step(closure)
    current_loss = closure().item()
    #optimizer.step()
    print(f"Iter {iters}: loss = {current_loss:.6f}")
    #print(f"Iter {iters}: loss = {grad_loss:.6f}")

    if iters % 2 == 0:
        history.append(tt(dummy_data[0].detach().cpu()))


plt.figure(figsize=(12, 8))
for i in range(29):
    plt.subplot(3, 10, i + 1)
    plt.imshow(history[i])
    plt.title(f"iter= {i*2}")
    plt.axis('off')
plt.subplot(3,10, 30)
plt.imshow(tt(gt_data[0].cpu()))
plt.title("Ground Truth")
plt.axis('off')
plt.tight_layout()
plt.show()
