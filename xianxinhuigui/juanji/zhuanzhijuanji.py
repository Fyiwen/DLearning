import torch
from torch import nn
from d2l import torch as d2l

def trans_conv(X, K): # 按定义，自定义做反卷积的函数
    h, w = K.shape
    Y = torch.zeros((X.shape[0] + h - 1, X.shape[1] + w - 1)) # 初始化输出的形状
    for i in range(X.shape[0]):
        for j in range(X.shape[1]): # 遍历输入的每一个元素
            Y[i: i + h, j: j + w] += X[i, j] * K # 每一个计算到的新值叠加到之前几轮计算产生的输出的对应位置上
    return Y
X = torch.tensor([[0.0, 1.0], [2.0, 3.0]]) # 定义了一个2*2的输入
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]]) # 2*2的核
print(trans_conv(X, K))

# 这里reshape成4维的x、k，更适合实际应用，通过调用高级API获得反卷积结果
X, K = X.reshape(1, 1, 2, 2), K.reshape(1, 1, 2, 2) # (批次，通道，高，宽)
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, bias=False) # 输入通道，输出通道
tconv.weight.data = K # 将上面定义的K内元素，作为这个反卷积层运算时采用的参数，一般情况下不用这个，用默认的，但是这里为了和上面那个写法得到一样结果，所以这样
print(tconv(X))


# 当将高和宽两侧的填充数指定为1时，转置卷积的输出中将删除第一和最后的行与列
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, padding=1, bias=False)
tconv.weight.data = K
print(tconv(X))

tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2, bias=False)
tconv.weight.data = K
print(tconv(X))

# 验证对于多个输入和输出通道时，转置卷积与常规卷积以相同方式运作
X = torch.rand(size=(1, 10, 16, 16)) # 随机生成一个16*16的10输入通道的输入
conv = nn.Conv2d(10, 20, kernel_size=5, padding=2, stride=3) # 输入进入卷积层，输入通道10个，每个输入通道分配一个5*5，20个输出通道每个分配10*5*5
tconv = nn.ConvTranspose2d(20, 10, kernel_size=5, padding=2, stride=3) # 再进入反卷积
print(tconv(conv(X)).shape == X.shape)


# 使用corr2d函数计算卷积输出Y
X = torch.arange(9.0).reshape(3, 3)
K = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
Y = d2l.corr2d(X, K)
print(Y)

# 使用矩阵乘法，计算实现卷积
def kernel2matrix(K):# 这个函数用于将卷积核K重写为包含大量0的稀疏权重矩阵W，只有使用这样定义出来的w才能获得和上面一样的结果
    k, W = torch.zeros(5), torch.zeros((4, 9))
    k[:2], k[3:5] = K[0, :], K[1, :] # k=【1，2，0，3，4】。。。。K[0, :],K中第0行所以元素1，2.
    W[0, :5], W[1, 1:6], W[2, 3:8], W[3, 4:] = k, k, k, k # W[0, :5]w中第0行前五个放k中内容，其他都是0
    return W
W = kernel2matrix(K) # 利用上面定义的2*2的K，产生一个4*9稀疏权重矩阵
print(W)
print(Y == torch.matmul(W, X.reshape(-1)).reshape(2, 2)) #X.reshape(-1)：逐行连结输入X，获得了一个长度为9的矢量。 然后，W的矩阵乘法和向量化的X给出了一个长度为4的向量。 reshape它之后，可以获得与上面的原始卷积操作所得相同的结果Y。

# 同样，我们可以使用矩阵乘法来实现转置卷积
Z = trans_conv(Y, K) # 直接用
print(Z == torch.matmul(W.T, Y.reshape(-1)).reshape(3, 3)) # 判断用矩阵乘法产生的结果和直接做的结果是不是一样













