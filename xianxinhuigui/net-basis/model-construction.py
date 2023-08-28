import os

import torch
from torch import nn
from torch.nn import functional as F

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 1.实例化nn.Sequential来构建我们的模型,一个简单的多层感知机
net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(),
                    nn.Linear(256, 10))# 包含一个具有256个单元和ReLU激活函数的全连接的隐藏层，然后是一个具有10个隐藏单元且不带激活函数的全连接的输出层。层的执行顺序就是传入参数的顺序，即一层线性层一层relu一层线性层
X = torch.rand(2, 20)
print('1.实例化nn.Sequential来构建我们的模型')
print(net(X))  # 通过net(X)调用我们的模型来获得模型的输出。是net.__call__(X)的简写


# 2.自定义模型
class MLP(nn.Module):
    def __init__(self):  # # 调用`MLP`的父类的构造函数来执行必要的初始化。
        # 这样，在类实例化时也可以指定其他函数参数，例如模型参数`params`（稍后将介绍）
        super().__init__()
        self.hidden = nn.Linear(20, 256)  # 隐藏层
        self.out = nn.Linear(256, 10)  # 输出层

    def forward(self, X):  # 正向传播（forward）函数：将列表中的每个块连接在一起，将每个块的输出作为下一个块的输入。即如何根据输入`X`返回所需的模型输出
        return self.out(F.relu(self.hidden(X)))


net = MLP()
print('2.自定义模型')
print(net(X))  # 前边说调用net() 就相当于调用net.__call__(X)，因为我们在自己的MLP中写了forward，但是我们没有调用，只使用net() 他就自动执行forward了。就是因为会自动调用.__call__函数使forward执行


# 3.自定义顺序模型,即模仿Sequential，Sequential设计是为了把其他模块串起来
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        print(args)  # 一个tuple, tuple和list非常类似，但是，tuple一旦创建完毕，就不能修改了。
        for block in args:
            # 这里，`block`是`Module`子类的一个实例。我们把它保存在'Module'类的成员变量
            # `_children` 中。`block`的类型是OrderedDict。
            self._modules[block] = block  # 每个Module都有一个_modules属性

    def forward(self, X):  # 正向传播函数，用于将输入按追加块的顺序传递给块组成的“链条”
        # OrderedDict保证了按照成员添加的顺序遍历它们
        print(self._modules.values())
        for block in self._modules.values():
            X = block(X)
        return X

print('3.自定义顺序模型')
net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))  # MySequential类提供了与默认Sequential类相同的功能
print(net(X))


# 4.如何将任意代码集成到神经网络计算的流程中
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.rand_weight = torch.rand((20, 20), requires_grad=True)
        self.liner = nn.Linear(20, 20)

    def forward(self, X):
        X = self.liner(X)
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        X = self.liner(X)
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()


print('4.如何将任意代码集成到神经网络计算的流程中')
net = FixedHiddenMLP()
print(net(X))


# 5.组合块
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))

print('5.组合块')
chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
print(chimera(X))