import torch
from torch import nn
from d2l import torch as d2l

def pool2d(X, pool_size, mode='max'):  # 此函数实现池化层的操作，输出为输入中每个区域的最大值或平均值。这里是默认不给参数时，以最大池化层作操作
    p_h, p_w = pool_size  # 得到池化框区域的宽高
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))  # 初始化输出结果的尺寸
    for i in range(Y.shape[0]):  # 做遍历
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()  # 以最大池化层得到结果，即选择每一次范围内最大的结果
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()  # 以平均池化层得到结果，即每一次选择范围内的平均数作为结果
    return Y

X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
pool2d(X, (2, 2))  # 没给参数，所以默认以最大池化层操作
pool2d(X, (2, 2), 'avg')  # 给了参数，以平均池化层操作

# 用深度学习框架中内置的二维最大池化层，来演示池化层中填充和步幅的使用。
X = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))  # 构造了一个输入张量X，它有四个维度，其中样本数和通道数都是1
print(X)

# 默认情况下，深度学习框架中的步幅与池化窗口的大小相同。 因此，如果我们使用形状为(3, 3)的汇聚窗口，那么默认情况下，我们得到的步幅形状为(3, 3)
pool2d = nn.MaxPool2d(3)  # 使用框架构造了(3, 3)的汇聚窗口，步幅没有声明所以按默认来
pool2d(X)  # 经过最大池化层后的输出

pool2d = nn.MaxPool2d(3, padding=1, stride=2)  # 使用框架构造了(3, 3)的汇聚窗口，填充和步幅手动设定
pool2d(X)

pool2d = nn.MaxPool2d((2, 3), stride=(2, 3), padding=(0, 1))  # 设定一个任意大小的矩形汇聚窗口，并分别设定填充和步幅的高度和宽度
pool2d(X)

# 在处理多通道输入数据时，池化层在每个输入通道上单独运算，而不是像卷积层一样在通道上对输入进行汇总。 所以池化层的输出通道数与输入通道数相同。
X = torch.cat((X, X + 1), 1)  # 在通道维度上连结张量X和X + 1，以构建具有2个通道的输入
print(X)
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X)  # 可以看到池化后输出通道的数量仍然是2







