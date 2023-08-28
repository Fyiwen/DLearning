import os
import torch
from torch import nn
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 本篇有关参数的处理，是因为在训练的过程中，我们的目标就是找到让损失函数最小化的参数值。经过训练之后我们需要将这些参数拿出来做预测，或者在其他地方使用，所以为了便于使用，做出如下介绍
# 访问参数，用于调试、诊断和可视化。
# 参数初始化。
# 在不同模型组件间共享参数
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))  #简易实现了一个多层感知机，然后弄了一个X做输入
X = torch.rand(size=(2, 4))
print(net(X))  # 当通过Sequential类定义模型时，我们可以通过索引来访问模型的任意层

print(net[0])  # 我们可以通过前边的序号得到想要的层，# 当通过Sequential类定义模型时，我们可以通过索引来访问模型的任意层
print(net[1])

print('1.访问第二个全连接层的参数')  # 相应的层序号+方法调用，提取网络的偏置或参数
print(net[2].state_dict())  # 这个函数是输出此层所有参数
print(net[2].bias)  # 第二个神经网络层提取偏置
print(net[2].bias.data)  # 第二个神经网络层提取偏置的实际值
print(net[2].weight.grad is None)  # 第二个神经网络层提取偏置的梯度，由于我们还没有调用这个网络的反向传播，所以参数的梯度处于初始状态。

print('2.一次性访问所有参数')  # 这里*是一个解包器 ，用于输出列表的每一个元素
print(*[(name, param.shape) for name, param in net[0].named_parameters()])  # 输入层的参数，第一个是解包net的第0层的参数参数名称和参数形状，直接调用net.named_parameters()：会把所有层的参数打印出来
print(*[(name, param.shape) for name, param in net.named_parameters()])  # 第二个是解包net所有层的参数名称和参数形状
print(*net.named_parameters(),end="\n",sep='\n')  # 第三个是解包net的参数列表

# 还可以这样获取参数列表
print(net.state_dict()['2.bias'].data)
print(net.state_dict()['0.weight'])  # 后边不管加不加.data都可以直接输出参数的值



# 3.嵌套块的参数
def block1():  # 首先定义一个生成块的函数（可以说是块工厂），然后将这些块组合到更大的块中
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4), nn.ReLU())


def block2():  # 将这些块组合到更大的块中
    net = nn.Sequential()
    for i in range(4):  # 在这里嵌套
        net.add_module(f'block{i}', block1())
    return net


print('3.嵌套块的参数')
rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
print(rgnet)  # 输出一下这个网路
print(rgnet(X))
print(rgnet[0][1][0].bias.data)  # 访问第一个主要的块，其中第二个子块的第一层的偏置项，因为层是分层嵌套的，我们也可以像通过嵌套列表索引一样访问它
print(rgnet.state_dict()['0.block 1.0.bias']) # 首先指明在哪一个块。再指明哪一个块上的哪一层

# 4.1内置的初始化器
def init_normal(m):  # 调用内置的初始化器。
    if type(m) == nn.Linear:  # m就是一个module
        nn.init.normal_(m.weight, mean=0, std=0.01)  #  # 给权重赋值-将所有权重参数初始化为标准差为0.01的正态分布
        nn.init.zeros_(m.bias)  #  # 给偏置赋值-将偏置设为0


print('4.1内置的初始化器')
net.apply(init_normal)  # 将net里面所以层遍历一遍
print(net[0].weight.data[0], net[0].bias.data[0])


# 4.2所有参数初始化为给定的常数
def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)


print('4.2所有参数初始化为给定的常数')
net.apply(init_constant)
print(net[0].weight.data[0], net[0].bias.data[0])

#还可以不同的层运用不同的初始化方法
net[0].apply(init_normal)
net[2].apply(init_constant)
net[4].apply(init_constant)

print(net[0].weight.data)
print(net[2].weight.data)


# 4.3使用Xavier初始化方法初始化第一层，然后第二层初始化为常量值42
def xavier(m):  # Xavier初始化方法
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


def init_42(m):  # 初始化为常量值42的方法
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)


print('4.3使用Xavier初始化方法初始化第一层，然后第二层初始化为常量值42')
net[0].apply(xavier)
net[2].apply(init_42)
print(net[0].weight)
print(net[2].weight.data)


# 5.参数自定义初始化，除了初始化为常数，初始化为高斯分布，pytorch还支持其他初始化方法，但是如果你说这些就是不能满足我的要求，我需要一个其他的初始化方法，那也可以自己写
def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape) for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)  # 将w全部初始化为（-10,10）的均匀分布
        m.weight.data *= m.weight.data.abs() >= 5  # 进行判定，看每一个权重的绝对值是否大于等于5，如果大于等于5则证明在(5, 10)和(-10，-5)区间上，那返回true，也就是1，m.weight.data乘1数值不变；反之会返回false，也就是0，将m.weight.data置零


print('5.参数自定义初始化')
net.apply(my_init)
print(net[0].weight[:2])

print(net[4].weight.data) #我们还可以手动修改某个参数的值
net[4].weight.data[:] += 10
print(net[4].weight.data)

net[4].weight.data[0, 0] = 666
print(net[4].weight.data)

# 6.多个层间共享参数，参数绑定
# 我们需要给共享层一个名称，以便可以引用它的参数。
shared = nn.Linear(8, 8)  # 设置共享层
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), shared, nn.ReLU(), shared,
                    nn.ReLU(), nn.Linear(8, 1))  # 定义网络
net(X)  # 调用网络处理X
# 检查参数是否相同
print('6.多个层间共享参数')
print(net[2].weight.data[0] == net[4].weight.data[0])  # 确保它们实际上是同一个对象，而不只是有相同的值
net[2].weight.data[0, 0] = 100  # 这里看似修改了一个，但是修改的是share对应的内容，所以另一个也变
# 我们需要给共享层一个名称，以便可以引用它的参数。
print(net[2].weight.data[0] == net[4].weight.data[0])