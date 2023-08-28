# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
# 生成数据集，直接用了库（为判断预测准确性这里用确定的参数来生成数据集，便于直观与数据集自己训练后预测的参数比较）
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

# 批量读取数据集
def load_array(data_arrays, batch_size, is_train=True):  # @save,true表示希望数据迭代器对象在每个迭代周期内打乱数据，迭代器是一个可以记住遍历位置的对象。迭代器对象从集合的第一个元素开始访问，直到所有的元素被访问完结束。迭代器只能往前不能后退。迭代器是Python中的容器类的数据类型，可以同时存储多个数据，取迭代器中的数据只能一个一个地取，而且取出来的数据在迭代器中就不存在了。
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)  #整理数据格式，即从数据集中获取全部
    return data.DataLoader(dataset, batch_size, shuffle=is_train)  # 根据批量大小，分批次向模型中传入数据，这里shuffle就是默认true，需要打乱顺序

batch_size = 10
data_iter = load_array((features, labels), batch_size)  # 按数据集和确定批量大小读取数据存入此变量中
print(next(iter(data_iter)))  # 为检验是否正常工作，next从python迭代器中获取第一批读取的数据来看看，iter生成迭代器

from torch import nn
# 定义模型，使用框架的预定义好的层。只需关注使用哪些层来构造模型，而不必关注层的实现细节，一个模型变量net，它是一个Sequential类的实例。 Sequential类将多个层串联在一起。 当给定输入数据时，Sequential实例将数据传入到第一层，然后将第一层的输出作为第二层的输入，以此类推
# 这里线性只有一个单层，被称为全连接层，全连接层在Linear类中定义
net = nn.Sequential(nn.Linear(2, 1)) # 2是指定输入特征形状，1指定输出特征形状

# 初始化模型参数
net[0].weight.data.normal_(0, 0.01)  # net[0]访问网络中第一个图层，然后weight.data访问这个层中参数为他们初始化，用normal和fill重写参数值，w从均值0，标准差0.01中随机取样
net[0].bias.data.fill_(0)  # b初始化为0

# 定义损失函数，这里用的平方L2范数，一般默认返回所有样本损失的平均值
loss = nn.MSELoss()

#优化算法,以便一步步让参数最优化，还是用小批量随机梯度下降
trainer = torch.optim.SGD(net.parameters(), lr=0.03)  # 调用SGD即小批量随机梯度下降优化方法，我们要指定需优化的参数（可通过net.parameters()从我们的模型中获得）以及优化算法所需的超参数。小批量随机梯度下降只需要设置lr值，这里设置为0.03。

#训练
num_epochs = 3  #用全部数据训练三遍
for epoch in range(num_epochs):
    for X, y in data_iter:  # 三遍中每一轮所有数据按批次被处理，经几轮处理完，X里面放特征，y放标签
        l = loss(net(X) ,y)  # 调用net(X)生成预测y^，与后面y一起计算损失放入l
        trainer.zero_grad()  # 优化器梯度清零
        l.backward()
        trainer.step()  # 调用优化算法来更新模型参数。
    l = loss(net(features), labels)  # 一遍结束，算一次性全部预测完一遍的损失
    print(f'epoch {epoch + 1}, loss {l:f}')

# 比较生成数据集的真实参数和通过有限数据训练获得的模型参数。 要访问参数，我们首先从net访问所需的层，然后读取该层的权重和偏置
w = net[0].weight.data  # 预测的w值
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data  # 预测的b值
print('b的估计误差：', true_b - b)