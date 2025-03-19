import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# Define CNN model for CIFAR-10
class SimpleCNN(nn.Module):
    def __init__(self, seed=42):
        super(SimpleCNN, self).__init__()
        torch.manual_seed(seed)
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_param_dict(self):
        """Get the model parameters as a dictionary."""
        return {k: v.clone() for k, v in self.state_dict().items()}

    def set_param_dict(self, params_dict):
        """Set the model parameters from a dictionary."""
        self.load_state_dict(params_dict)