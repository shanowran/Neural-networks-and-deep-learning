"""
@Time ： 2022/10/27 18:59
@Auth ： Luminous
@Contact:  3199409618@qq.com
@File ：define_restnet.py
@IDE ：PyCharm
"""
import torch
import torch.nn.functional as F
import torch.nn as nn


class blocks(nn.Module):
    def __init__(self, in_chan, out_chan, stride=1):
        super(blocks, self).__init__()
        self.conv1 = nn.Conv2d(in_chan, out_chan, 3, stride, padding=1)
        self.bin1 = nn.BatchNorm2d(out_chan)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_chan, out_chan, 3, stride=1, padding=1)
        self.bin2 = nn.BatchNorm2d(out_chan)
        self.extra = nn.Sequential()
        if in_chan != out_chan:
            self.extra = nn.Sequential(
                nn.Conv2d(in_chan, out_chan, 1, stride=2, padding=0),
                nn.BatchNorm2d(out_chan)
            )

    def forward(self, x):
        y = x
        x = self.conv1(x)
        x = self.bin1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bin2(x)
        x = self.extra(y) + x
        return x


class Resnet(nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()
        # input->(3, 224, 224)
        # output->(64, 56, 56)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        )
        # output->(64, 56, 56)
        self.blk1 = blocks(64, 64, 1)
        # output->(64, 56, 56)
        self.blk2 = blocks(64, 64, 1)
        # output->(128, 28, 28)
        self.blk3 = blocks(64, 128, 2)
        # output->(128, 28, 28)
        self.blk4 = blocks(128, 128, 1)
        # output->(256, 14, 14)
        self.blk5 = blocks(128, 256, 2)
        # output->(256, 14, 14)
        self.blk6 = blocks(256, 256, 1)
        # output->(512, 7, 7)
        self.blk7 = blocks(256, 512, 2)
        # output->(512, 7, 7)
        self.blk8 = blocks(512, 512, 1)
        # output->(512, 1, 1)
        self.avgpool = nn.AvgPool2d(7, 1)
        # output->(10)
        self.liner = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)
        x = self.blk5(x)
        x = self.blk6(x)
        x = self.blk7(x)
        x = self.blk8(x)
        x = self.avgpool(x)
        x = x.view(-1, 512)
        x = self.liner(x)
        return x





def main():
    x = torch.randn(32, 3, 224, 224)
    net = Resnet()
    out = net(x)
    print(out)


if __name__ == '__main__':
    main()





