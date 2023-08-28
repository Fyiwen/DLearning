import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256 # 这里批量大小256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)  # 获取fashionmnist数据集，返回数据迭代器到这两个变量对应中，一个作为训练集一个作为测试集

# 定义模型
# PyTorch不会隐式地调整输入的形状。因此，
# 我们在线性层前定义了展平层（flatten），来调整网络输入的形状，Flatten()将任何维度的tensor转化为2D的tensor，这里softmax其实用不到，但因为多层感知机和他的实现差不多，所以为了模板化写上，在多层感知机里，有了这个层输入的3D内容就会被展平成图片再送去作为后面一层的输入
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))  # 784个输入，因为选的数据集十个类别，所以这是10个输出的全连接层

# 初始化模型参数
def init_weights(m):  # 相当于m相当于模型中的某一层，这里只需要给全连接层初始化一下参数
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)  # 给定了方差，从正态分布中取值为参数初始化

net.apply(init_weights)  #apply()：将init_weights函数运用到模型的每个层中来初始化模型参数

#损失函数交叉熵
loss = nn.CrossEntropyLoss(reduction='none')

# 优化算法，还是用小批量随机梯度下降
trainer = torch.optim.SGD(net.parameters(), lr=0.1)

#训练，调用从0实现中的训练函数来训练模型,这里已经封装在了d2l库里
num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
d2l.plt.show()