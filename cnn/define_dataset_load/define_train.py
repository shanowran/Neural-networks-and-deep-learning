"""
@Time ： 2022/10/28 14:18
@Auth ： Luminous
@Contact:  3199409618@qq.com
@File ：define_train.py
@IDE ：PyCharm
"""
import torch
import torch.nn as nn
import define_restnet
import datasets
import define_restnet as resnet
import torch.optim as optim
import torch.utils.data.dataloader as Dataloader
import torch.nn.functional as F


def main():
    root ="D:\卷积神经网络\data\pokeman"
    device = "cuda"
    db = datasets.numberdatasets(root, 224, "train")
    db_test = datasets.numberdatasets(root, 224, "var")
    test_loader = Dataloader.DataLoader(db_test, batch_size=32, shuffle=True, num_workers=2)
    net = resnet.Resnet().to(device)
    criter = nn.CrossEntropyLoss().to(device)
    optims = optim.Adam(net.parameters(), lr=1e-3)
    loader = Dataloader.DataLoader(db, batch_size=32, shuffle=True, num_workers=2)
    for epoch in range(10):
        loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optims.zero_grad()
            out = net(x)
            print(out.size())
            losses = criter(out, y)
            losses.backward()
            optims.step()
            loss = loss + losses.item()
        print("loss:",loss)
        num_image = 0
        num_correct = 0
        with torch.no_grad():
            for xx, yy in test_loader:
                xx, yy = xx.to(device), yy.to(device)
                output = net(xx)
                pre = output.argmax(dim=1)
                num_image += xx.size(0)
                num_correct += torch.eq(pre, yy).float().sum().item()
            print("epoch:%d correct:%f"%(epoch, num_correct / num_image))


if __name__ == '__main__':
    main()
