import torch
import torch.nn as nn
from torch.nn import functional as F


class letnet5(nn.Module):
    def __init__(self):
        super(letnet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d((2, 2), 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d((2, 2), 2)
        self.liner1 = nn.Linear(400, 120)
        self.liner2 = nn.Linear(120, 84)
        self.liner3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = F.relu(x)
        x = x.view(-1, 400)
        x = self.liner1(x)
        x = self.liner2(x)
        x = self.liner3(x)
        return x


# net = letnet5()
# if torch.cuda.is_available():
#     x = torch.randn(32, 3, 32, 32)
#     out = net(x)
#     # out = F.softmax(x, 1)
#     print(out)

