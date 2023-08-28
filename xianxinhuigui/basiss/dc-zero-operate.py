import os
import sys

import torch
from d2l import torch as d2l
from torch import nn

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 初始化模型参数
"""
Fashion-MNIST中的每个图像由所有图像共分为10个类别。忽略像素之间的空间结构，我们可以将每个图像视为具有784个输入特征和10个类的简单分类数据集
"""
"""
首先，我们将实现一个具有单隐藏层的多层感知机， 它包含256个隐藏单元。 注意，我们可以将这两个变量都视为超参数。 通常，我们选择2的若干次幂作为层的宽度。 因为内存在硬件中的分配和寻址方式，这么做往往可以在计算上更高效。

我们用几个张量来表示我们的参数。 注意，对于每一层我们都要记录一个权重矩阵和一个偏置向量。 跟以前一样，我们要为损失关于这些参数的梯度分配内存。
"""
num_inputs, num_outputs, num_hiddens = 784, 10, 256  # 输入，输出，隐藏层大小

W1 = nn.Parameter(torch.randn(  # nn.Parameter()函数的目的就是让该变量在学习的过程中不断的修改其值以达到最优化。torch.randn()：返回一个张量，包含了从标准正态分布中抽取的一组随机数，randn()：第一个参数是行数（输入数），第二个参数是列数（隐藏层大小）
    num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))  # zeros()：偏差，是一个向量-大小为隐藏层的大小，默认设为0
W2 = nn.Parameter(torch.randn(
    num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [W1, b1, W2, b2]  # # W1，b1：第一层 # W2, b2：第二层（隐藏层）

# 激活函数
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)

# 模型
"""
因为我们忽略了空间结构， 所以我们使用reshape将每个二维图像转换为一个长度为num_inputs的向量
"""
def net(X):
    X = X.reshape((-1, num_inputs))  # 第一步，先把图片拉成一个矩阵
    H = relu(X@W1 + b1)  # 这里“@”代表矩阵乘法
    return (H@W2 + b2)

# 损失函数交叉熵损失
loss = nn.CrossEntropyLoss(reduction='none')

#训练多层感知机的训练过程与softmax回归的训练过程完全相同。 可以直接调用d2l包的train_ch3函数（参见 3.6节 ）， 将迭代周期数设置为10，并将学习率设置为0.1.
num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)

# 稍微评估一下学习的模型，在一些测试数据上应用这个模型
d2l.predict_ch3(net, test_iter)









