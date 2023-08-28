import torch
from torch import nn
from d2l import torch as d2l

def corr2d(X, K):
    """计算二维互相关运算的函数"""
    h, w = K.shape  # K是卷积核，这里调出了他的行数列数
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))  # 初始化输出矩阵Y大小
    for i in range(Y.shape[0]):  # 行
        for j in range(Y.shape[1]): # 列
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()  # 互相关运算，双重嵌套，给输出矩阵赋值
    return Y
# 调用函数计算看看
X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
corr2d(X, K)


# 卷积层的定义
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))  # torch.rand：均匀分布，从区间[0, 1)中抽取的一组随机数（返回一个张量）
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias  # 调用上面的函数完成输出


# 如下是卷积层的一个简单应用：通过找到像素变化的位置，来检测图像中不同颜色的边缘。
X = torch.ones((6, 8))
X[:, 2:6] = 0  # 首先，我们构造一个像素的黑白图像。中间四列为黑色（0），其余像素为白色（1）
print(X)
K = torch.tensor([[1.0, -1.0]])  # 构造一个高度为1、宽度为2的卷积核K,当进行互相关运算时，如果水平相邻的两元素相同，则输出为零，否则输出为非零
Y = corr2d(X, K)  # 现在，我们对参数X（输入）和K（卷积核）执行互相关运算。
print(Y)  # 输出Y中的1代表从白色到黑色的边缘，-1代表从黑色到白色的边缘，其他情况的输出为0

# Y=corr2d(X.t(), K)  # 现在我们将输入的二维图像转置，再进行如上的互相关运算。 其输出如下，之前检测到的垂直边缘消失了。 不出所料，这个卷积核K只可以检测垂直边缘，无法检测水平边缘。
# print(Y)

# 学习卷积核，仅查看“输入-输出”对来学习由X生成Y的卷积核
# 先构造一个卷积层，并将其卷积核初始化为随机张量。接下来，在每次迭代中，我们比较Y与卷积层输出的平方误差，然后计算梯度来更新卷积核。为了简单起见，我们在此使用内置的二维卷积层，并忽略偏置。
conv2d = nn.Conv2d(1,1, kernel_size=(1, 2), bias=False)  # 构造一个二维卷积层，它具有1个输入通道，1个输出通道和形状为（1，2）的卷积核

# 这个二维卷积层使用四维输入和输出格式（批量大小、通道、高度、宽度），
# 其中批量大小和通道数都为1
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))
lr = 3e-2  # 学习率

for i in range(10):  # 在每次迭代中，比较Y与卷积层输出的平方误差，然后计算梯度来更新卷积核
    Y_hat = conv2d(X)  # 输出值
    l = (Y_hat - Y) ** 2
    conv2d.zero_grad()
    l.sum().backward()
    # 迭代卷积核
    conv2d.weight.data[:] -= lr * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        print(f'epoch {i+1}, loss {l.sum():.3f}')

conv2d.weight.data.reshape((1, 2))  # 在次迭代之后，误差已经降到足够低。现在我们来看看我们所学的卷积核的权重张量