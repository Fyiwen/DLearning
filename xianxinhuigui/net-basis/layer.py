import os

import torch
from torch import nn
from torch.nn import functional as F

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 自定义层
# 之前接触的楼层是什么。比如nn.Linear,nn.ReLU等。他们的作用就是作为某一层的处理。他们两个的区别在于前者有参数，后者是没有参数列表的。那现在我们也来实现一些有参数和没有参数列表的层操作


# 不带参数的层
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):  # 我们也只需要定义前向传播就可以了。这个自建一层的作用是让每一个特征量都减去其平均值
        return X - X.mean()


print('1.不带参数的层')
layer = CenteredLayer()
print(layer(torch.FloatTensor([1, 2, 3, 4, 5])))

net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())  # 这个模型其实并不复杂，它只有两层。第一个是一个线性层。第二个就是我们的自定义层
print(net)
Y = net(torch.rand(4, 8))  # Y是4*128维的，生成一组随机的测试数据Y。然后使用我们构建的网络对外进行计算，然后输出其结果的平均值。不出意外结果应该是0。虽然这里显示的不是0。这是因为浮点数的存储精度问题，你当然可以把这个极小的数近似看作它是0
print(Y.mean())


# 带参数的层
class MyLinear(nn.Module):  # 自定义实现了一个全链接层。这个层里的参数需要用到权重和偏置，在计算之后最后返回再使用ReLU激活函数
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units))

    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)


print('2.带参数的层')
dense = MyLinear(5, 3)
print(dense.weight)

Y = dense(torch.rand(2, 5))
print(Y)

net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
print(net(torch.rand(2, 64)))