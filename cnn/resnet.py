import torch
import torch.nn as nn
import  torch.nn.functional as F


class Block(nn.Module):
    """
    短接模块
    """
    def __init__(self, in_chan, out_chan, stride=1):
        """
        :param in_chan:
        :param out_chan:
        """
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_chan, out_chan, (3, 3), stride=stride, padding=1)
        self.bat = nn.BatchNorm2d(out_chan)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_chan, out_chan, 3, stride=1, padding=1)
        self.bat2 = nn.BatchNorm2d(out_chan)
        self.extra = nn.Sequential()
        if in_chan != out_chan:
            self.extra = nn.Sequential(
                nn.Conv2d(in_chan, out_chan, 1, stride=stride),
                nn.BatchNorm2d(out_chan)
            )

    def forward(self, x):
        """

        :param x:
        :return:
        """
        y = x
        # print('y:', y.size())
        x = self.conv1(x)
        # print('x:', x.size())
        x = self.bat(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bat2(x)
        # print(x.size())
        # x是8 y是32 报错
        # print(self.extra(y).size())
        x = self.extra(y) + x
        # return 0
        return x


class Resnet(nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()
        self.Conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64)
        )
        self.blk1 = Block(64, 128, 2)
        self.blk2 = Block(128, 256, 2)
        self.blk3 = Block(256, 512, 2)
        self.blk4 = Block(512, 1024, 2)
        self.linear = nn.Linear(1024*2*2, 10)

    def forward(self, x):
        x = self.Conv1(x)
        x = F.relu(x)
        x = self.blk1(x)
        x = F.relu(x)
        x = self.blk2(x)
        x = F.relu(x)
        x = self.blk3(x)
        x = F.relu(x)
        x = self.blk4(x)
        x = F.relu(x)
        x = x.view(-1, 1024*2*2)
        x = self.linear(x)
        # x = F.relu(x)
        return x


# net = Resnet()
# x = torch.randn(1, 3, 32, 32)
# out = net(x)
# print(out.size())
# print(out)
