import os

import torch
from torch import nn
from torch.nn import functional as F

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 读写文件
#读写文件其实不是读取数据集。是当你的训练时要定期存储中间结果，以确保在服务器电源不小心被断掉，或者出现其他情况的时候，损失掉你前几天的计算结果。这一节要做的就是如何存储权重向量和整个模型。

# 加载和保存张量
# 对于单个张量，我们可以直接调用load和save函数分别读写它们
                         # torch.save(obj, f, pickle_module=<module 'pickle' from '.../pickle.py'>, pickle_protocol=2)
                         # obj – 保存对象，f - 字符串，文件名，pickle_module – 用于pickling元数据和对象的模块，pickle_protocol – 指定pickle protocal 可以覆盖默认参数
                         # torch.load(f, map_location=None, pickle_module=<module 'pickle' from '.../pickle.py'>)
                           # map_location – 一个函数或字典规定如何remap存储位置，pickle_module – 用于unpickling元数据和对象的模块 (必须匹配序列化文件时的pickle_module )
x = torch.arange(4)  # 初始化一个x，将x存储到当前文件夹下并命名为x-file
torch.save(x, 'x-file')
x2 = torch.load("x-file")  # 声明一个x2再从文件中读回来
print(x2)

# 存储一个张量列表，然后把它们读回内存
y = torch.zeros(4)
# 切片也可以
torch.save(y[:2],'x-file') # torch.save([x, y],'x-files')
x2, y2 = torch.load('x-file')
print(x2, y2)

#存储字典也可以
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
print(mydict2)

# 加载和保存模型参数
# 深度学习框架提供了内置函数来保存和加载整个网络。但是是保存模型的参数而不是保存整个模型
class MLP(nn.Module):  # 多层感知机
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)  # 生成一个net，用它计算X，并将其赋值给Y
print(Y)

torch.save(net.state_dict(), 'mlp.params')  # 将net的参数保存起来

clone_net = MLP()  # 生成一个net_也是多层感知机，
clone_net.load_state_dict(torch.load("mlp.params"))  # net的参数直接加载文件中的参数
clone_net.eval()  # net_.eval()是将模型的模式改为评价模式

Y_clone = clone_net(X)  # 将新网络赋值给Y_clone，可以看到Y_clone和Y是相同的
print(Y_clone)