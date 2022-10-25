"""
@Time ： 2022/10/19 17:10
@Auth ： Luminous
@Contact:  3199409618@qq.com
@File ：rnn.py
@IDE ：PyCharm
"""
import torch.nn as nn
import torch
import torch.nn.functional as F


class Rnn(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super(Rnn, self).__init__()
        self.hidden_size = hidden_size
        self.Rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=1
        )
        self.liner = nn.Linear(self.hidden_size, out_size)

    def forward(self, x, hidden_pre):
        out, h_pre = self.Rnn(x, hidden_pre)
        # print(out.size())
        # print(h_pre.size())
        out = out.contiguous().view(-1, self.hidden_size)
        out = self.liner(out)
        out = out.unsqueeze(dim=0)
        return out, h_pre


# x = torch.randn(3, 10, 50)    #input:(batchsize, seq, feature_num) -> 几句话， 一句话几个单词， 一个单词用几维向量表示
# print(x)
# hidden_pre = torch.zeros(1, 3, 10) #hidden_pre:(num_layers, batchsize, hidden_feature_num)
# net = Rnn(50, 10, 10)
# out, h_pre = net(x, hidden_pre)
# print("out:", out.size())
# print("h_pre:", h_pre.size())
