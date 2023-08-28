import torch
from d2l import torch as d2l

# 多输入通道互相关运算函数。即对每个通道执行互相关操作，然后将结果相加.2输入通道1输出通道
def corr2d_multi_in(X, K):
    # 先遍历“X”和“K”的第0个维度（通道维度），再把它们加在一起
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K))# 按照输入通道维度做遍历，每次拿出x和k做互相关操作，最后求和.对每个通道执行互相关操作，然后将结果相加

X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
               [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])  # 2*3*3
K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]]) # 2*2*2
corr2d_multi_in(X, K)

# 实现一个计算多个通道输出的互相关函数
def corr2d_multi_in_out(X, K):
    # 迭代“K”的第0个维度，每次都对输入“X”执行互相关运算。
    # 最后将所有结果都叠加在一起
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)

# 通过将核张量K与K+1（K中每个元素加1）和K+2连接起来，构造了一个具有3个输出通道的卷积核
K = torch.stack((K, K + 1, K + 2), 0)
print(K.shape)
# 我们对输入张量X与卷积核张量K执行互相关运算。现在的输出包含3个通道，第一个通道的结果与先前输入张量X和多输入单输出通道的结果一致。
corr2d_multi_in_out(X, K)


# 验证1*1卷积相当于全连接层。
def corr2d_multi_in_out_1x1(X, K):  # 多输入输出的互相关操作
    c_i, h, w = X.shape  # 得输入通道数ci
    c_o = K.shape[0]  # 得输出通道数co
    X = X.reshape((c_i, h * w))  # 把x拉长用作全连接层的输入
    K = K.reshape((c_o, c_i))  # k本来的尺寸是co*ci*1*1，现在也做调整作为全连接层的权重
    Y = torch.matmul(K, X)  # 全连接层中的矩阵乘法
    return Y.reshape((c_o, h, w))
# 当执行1*1卷积运算时，上述函数相当于先前实现的互相关函数corr2d_multi_in_out。让我们用一些样本数据来验证这一点。
X = torch.normal(0, 1, (3, 3, 3))
K = torch.normal(0, 1, (2, 3, 1, 1))

Y1 = corr2d_multi_in_out_1x1(X, K) # 这里是用全脸连接层实现方式来做1*1卷积核操作
Y2 = corr2d_multi_in_out(X, K)  # 这里是正常1*1卷积核的操作
assert float(torch.abs(Y1 - Y2).sum()) < 1e-6