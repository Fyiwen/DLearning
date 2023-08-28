import torch
from torch import nn
from d2l import torch as d2l

# 归一化函数
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):  #拉伸参数gamma、偏移参数beta,moving_mean,moving_var：全局均值和方差 ，eps：用于避免除零， momentum：用来更新全局均值和方差（0.9or0.1
    # 通过is_grad_enabled来判断当前模式是训练模式还是预测模式
    if not torch.is_grad_enabled():
        # 如果是在预测模式下（通过不计算梯度来判断），直接使用传入的移动平均所得的均值和方差
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:  # 在训练模式下，需要更新均值和方差
        assert len(X.shape) in (2, 4)  # 断言x的形状
        if len(X.shape) == 2:  # 使用全连接层的情况，计算特征维上的均值和方差,2是批量大小和特征
            mean = X.mean(dim=0)  # 因为是1*n的行向量，求每列的均值，相当于μ=1/m∑xi
            var = ((X - mean) ** 2).mean(dim=0)  #求方差，相当于σ2=1/m∑（xi-μ）2
        else:  # 使用二维卷积层的情况，计算通道维上（axis=1）的均值和方差。4是批量大小、输入输出通道、高、宽，                                                     这里我们需要保持X的形状以便后面可以做广播运算
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # 训练模式下，用当前的均值和方差做标准化
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # 更新移动平均的均值和方差
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # 加上缩放和移位，构成完整形态的公式，Y可以作为归一化处理完后的输出
    return Y, moving_mean.data, moving_var.data

class BatchNorm(nn.Module):  # 批量归一化层
    # num_features：完全连接层的输出数量或卷积层的输出通道数。
    # num_dims：2表示完全连接层，4表示卷积层
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:  # 初始化形状，便于下面变量的初始化，这是全连接层的形状
            shape = (1, num_features)
        else:  # 这是卷积层的形状
            shape = (1, num_features, 1, 1)
        # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成1和0
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # 非模型参数的变量初始化为0和1
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        # 如果X不在内存上，将moving_mean和moving_var复制到X所在显存上
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # 保存更新过的moving_mean和moving_var
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.9)
        return Y


# 使用批量规范化层的lenet，在激活函数前，卷积、全连接层之后加批量归一化层
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), BatchNorm(6, num_dims=4), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), BatchNorm(16, num_dims=4), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(16*4*4, 120), BatchNorm(120, num_dims=2), nn.Sigmoid(),
    nn.Linear(120, 84), BatchNorm(84, num_dims=2), nn.Sigmoid(),
    nn.Linear(84, 10))
# 训练
lr, num_epochs, batch_size = 1.0, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
                                                                                                                                                     #  net[1].gamma.reshape((-1,)), net[1].beta.reshape((-1,))#  看看从第一个批量规范化层中学到的拉伸参数gamma和偏移参数beta

"""
简单实现,不用自己定义批量化归一层
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), nn.BatchNorm2d(6), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.BatchNorm2d(16), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(256, 120), nn.BatchNorm1d(120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.BatchNorm1d(84), nn.Sigmoid(),
    nn.Linear(84, 10))
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
"""