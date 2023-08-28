import torch
from torch import nn

# 定义了一个计算卷积层的函数。并对输入和输出提高和缩减相应的维数
def comp_conv2d(conv2d, X):# 参数是卷积层和输入
    X = X.reshape((1, 1) + X.shape)  # 这里的（1，1）表示批量大小和通道数都是1，x会由二维变成四维
    Y = conv2d(X)
    return Y.reshape(Y.shape[2:])  # 经过reshape（）可以将信息从四维转为二维，省略前两个维度：批量大小和通道

# 这里每边上下左右都填充了1行或1列，因此总共添加了2行或2列
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)#定义了一个卷积层，并在所有侧边填充1个像素使用padding
X = torch.rand(size=(8, 8))  # 高度和宽度为8的输入
print(comp_conv2d(conv2d, X).shape)  # 可以看到则输出的高度和宽度也是8

conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1))  # 当卷积核的高度和宽度不同时，可以填充不同的高度和宽度，使输出和输入具有相同的高度和宽度。
print(comp_conv2d(conv2d, X).shape)


conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)  # 将高度和宽度的步幅设置为2，使用stride，可以将输入的高度和宽度减半
print(comp_conv2d(conv2d, X).shape)

conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
print(comp_conv2d(conv2d, X).shape)