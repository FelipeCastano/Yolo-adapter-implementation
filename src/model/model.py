import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 1)

    def forward(self, x):
        return self.fc(x)
