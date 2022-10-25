"""
@Time ： 2022/10/20 9:58
@Auth ： Luminous
@Contact:  3199409618@qq.com
@File ：RNNtest.py
@IDE ：PyCharm
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib as plt
import matplotlib.pyplot as pt
import torch.optim as optim
import rnn as rnn


# start = np.random.randint(3, size=1)[0] # 初始点，在0-3之间随机取数 是范围的起始
# print(start)
# time_steps = np.linspace(start, start+10, 50)   # start-start+10 范围内50个数
# data = np.sin(time_steps)
# x = np.linspace(0, 10, 100)
# y = 2 * x + 1
# y = np.sin(x)
# pt.plot(x, y)
# pt.plot(time_steps, data)
# pt.show()
# print(data)
# data = data.reshape(50, 1)
# print(data)
# x = torch.tensor(data[:-1]).float().view(1, 49, 1)
# y = torch.tensor(data[1:]).float().view(1, 49, 1)

net = rnn.Rnn(1, 50, 1)
ccriterion = nn.MSELoss()
optims = optim.Adam(net.parameters(), lr=1e-3)
for epoch in range(1000):
    start = np.random.randint(3, size=1)[0]  # 初始点，在0-3之间随机取数 是范围的起始
    time_steps = np.linspace(start, start + 10, 50)  # start-start+10 范围内50个数
    data = np.sin(time_steps)
    data = data.reshape(50, 1)
    x = torch.tensor(data[:-1]).float().view(1, 49, 1)  # 因为输入为第一个点到50个点，但是应与对比集合y大小相同，故为49
    # print("x:", x)
    y = torch.tensor(data[1:]).float().view(1, 49, 1)  # 因为预测为从第二个点到第51个点，输入应为从第一个点，到第50个点
    hidden = torch.zeros(1, 1, 50)
    out, hidden = net(x, hidden)
    # print("out:", out)
    hidden = hidden.detach()
    loss = ccriterion(out, y)
    net.zero_grad()
    loss.backward()
    optims.step()
    if epoch % 2 == 0:
        print("epoch:%d loss:%f "%(epoch, loss.item()))

    input = x[:, 0, :]  # input为[1,1]
    # print("input:", input)
    predictions = []
    # hidden1 = torch.zeros(1, 1, 50)
    # print(x.size(1))
    for j in range(x.size(1)):
        # input = x[:, j, :]
        input = input.view(1, 1, 1)
        # print("inpt:___",input)
        pred, hidden = net(input, hidden)  # 网络模型参数不能改变，要使用原先的网络预测数据，而不是新建网络
        input = pred
        # print(input)
        predictions.append(pred.detach().numpy().ravel()[0])
        # print("predict:", predictions)
    if epoch == 999:
        print("predict:", predictions)
        x = x.view(49, 1)
        x = x.numpy()
        time_steps = time_steps[1:]
        time_steps = time_steps.reshape(49, 1)
        fg2, ax2 = pt.subplots()
        ax2.plot(time_steps, x, marker=".")
        ax2.plot(time_steps, predictions, marker="o")
        pt.show()







