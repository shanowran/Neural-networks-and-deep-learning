import torch
import torch.nn as nn
from torch.utils import data
from torchvision import datasets
from torchvision.transforms import transforms
import torch.optim as optim
import letnet5 as letnet5
import resnet as resnet
import  torch.nn.functional as F


def main():
    batchsize = 32
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    train_set = datasets.CIFAR10('./data', train=True, download=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batchsize, shuffle=True, num_workers=2)
    test_set = datasets.CIFAR10('./data', train=False, transform=transform, download=False)
    test_loader = data.DataLoader(test_set, batch_size=batchsize, shuffle=True, num_workers=2)
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    print(labels.shape)
    print(images.shape)
    device = torch.device('cuda')
    # net = letnet5.letnet5().to(device=device)
    net = resnet.Resnet().to(device=device)
    print(net)
    loss = nn.CrossEntropyLoss().to(device)
    optims = optim.Adam(net.parameters(), lr=0.001)
    net.train()
    for epoch in range(0, 10):
        running_loss = 0.0

        for batchidx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optims.zero_grad()
            out = net(images)
            losses = loss(out, labels)
            losses.backward()
            optims.step()
            running_loss += losses.item()

        print(epoch, running_loss)

        total_num = 0
        total_ac = 0
        net.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                out = net(images)
                pre = out.argmax(dim=1)
                total_ac += torch.eq(pre, labels).float().sum().item()
                total_num += images.size(0)
            print(epoch, total_ac / total_num)
    # total_num1 = 0
    # total_ac1 = 0
    # net.eval()
    # with torch.no_grad():
    #     for i, data1 in enumerate(test_loader):
    #         images, labels = data1
    #         images, labels = images.to(device), labels.to(device)
    #         out = net(images)
    #         pre = out.argmax(dim=1)
    #         total_ac1 += torch.eq(pre, labels).float().sum().item()
    #         total_num1 += images.size(0)
    #     print(i, total_ac1 / total_num1)


if __name__ == '__main__':
    main()
