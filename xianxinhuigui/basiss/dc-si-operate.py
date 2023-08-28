import os
import sys

import torch
from d2l import torch as d2l
from torch import nn

#sys.path.append('D:\\pythonspace\\d2l\\d2lutil')  # 加入路径，添加目录
#import common

#os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 模型，添加了两个全连接层，第一层隐藏层包含256个隐藏单元，并使用relu激活函数，第二层是输出层。因为图片是一个3D的东西，然后使用nn.Flatten()为二维，nn.Linear(784, 256)线性层，输入为784，输出为256，nn.Linear(256, 10)线性层，输入为256，输出为10
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(),
                    nn.Linear(256, 10))

# 初始化模型参数，只要给全连接层的权值初始化就行，展平层不用
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)

#训练
batch_size, lr, num_epochs = 256, 0.1, 10
loss = nn.CrossEntropyLoss(reduction='none')  # 损失函数
trainer = torch.optim.SGD(net.parameters(), lr=lr)  # 优化算法
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size) # 下载测试数据集和训练数据集
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer) # 直接调用d2l包的训练函数train_ch3函数