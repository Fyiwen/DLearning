import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

class Residual(nn.Module):  #一个残差网络层，包含两个卷积层和一个旁路支路
    def __init__(self, input_channels, num_channels,use_1x1conv=False, strides=1):
        super().__init__()
        #  一个残差块包含两个卷积层，第一个卷积层通常改变输入输出通道数，并且改变输出的尺寸的形状大小，第二个卷积层输入输出通道数通常不会改变，每一个卷积层会跟着一个批量规范层
        self.conv1 = nn.Conv2d(input_channels, num_channels,kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,kernel_size=3, padding=1)
        if use_1x1conv:# 当输入与输出通道数不同后需要加一个1x1卷积层，来改变输入X的形状大小和通道数，以便后边与Y相加
            self.conv3 = nn.Conv2d(input_channels, num_channels,kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))# 输入x先经过卷积层，再经过归一化层，在经过激活函数
        Y = self.bn2(self.conv2(Y))  # 再经过第二个卷积层然后再经过归一化层
        if self.conv3: # 如果有用1*1卷积则输入需要经过此层，才能与Y相加
            X = self.conv3(X)
        Y += X  # 将输入X经过两层卷积层得到的输出Y与输入X相加后，再经过ReLU()激活函数,（必须保证X和Y的通道数和尺寸形状大小相同，才能相加）
        return F.relu(Y)
#输入和输出形状一致的情况
blk = Residual(3,3) #输入通道和输出通道都为3
X = torch.rand(4, 3, 6, 6)
Y = blk(X)
print(Y.shape)

# 我们也可以在增加输出通道数的同时，减半输出的高和宽
blk = Residual(3,6, use_1x1conv=True, strides=2)  # 输入和输出通道不一样了，所以要用1*1卷积
print(blk(X).shape)

# resnet网络
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))  # 第一个块

def resnet_block(input_channels, num_channels, num_residuals,first_block=False):                                                                # 相当于上面是一个小残差块，这个看书是把类似的小块组成大块，然后下面bi是很多个大块
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))  # 第二个块
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))

# resnet完整网络
net = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(), nn.Linear(512, 10))
#可以查看不同模块的输出形状
X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)
# 训练
lr, num_epochs, batch_size = 0.05, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())

