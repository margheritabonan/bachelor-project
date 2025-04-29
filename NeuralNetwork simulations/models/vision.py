import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad


def weights_init(m):
    if hasattr(m, "weight"):
        m.weight.data.uniform_(-0.5, 0.5)
    if hasattr(m, "bias"):
        m.bias.data.uniform_(-0.5, 0.5)
        
class LeNet(nn.Module):
    def __init__(self, in_channels = 1):
        super(LeNet, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
            act(),
        )
        self.fc =  None 
        
    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)  
        if self.fc is None:
            self.fc = nn.Sequential(
                nn.Linear(out.size(1), 100)  # use the flattened size as input
            ).to(x.device)
        out = self.fc(out)
        return out


