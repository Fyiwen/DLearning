import math
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l

# 人工生成数据集，用多项式y=5+1.2x-3.4x^2/2!+5.6x^3/3!+зwhereз~N(0,0.1^2),噪声项服从均值为0且标准差为0.1的正态分布。                                   在优化的过程中，我们通常希望避免非常大的梯度值或损失值。 这就是我们将特征从调整为/!的原因， 这样可以避免很大的带来的特别大的指数值。 我们将为训练集和测试集各生成100个样本，高斯噪声就是服从高斯分布的随机误差
max_degree = 20  # 多项式的最大阶数，因为是讨论过拟合欠拟合的问题，这里设置这个为20，表示最大可以用20阶来拟合
n_train, n_test = 100, 100  # 训练和测试数据集大小
true_w = np.zeros(max_degree)  # 分配大量的空间
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])

features = np.random.normal(size=(n_train + n_test, 1))# 随机生成x
np.random.shuffle(features)  # 将x打乱顺序
poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))  #对features做乘方，即x的多少阶
for i in range(max_degree):
    poly_features[:, i] /= math.gamma(i + 1)  # gamma(n)=(n-1)!，现在poly里存的是x的阶/!的结果
# labels的维度:(n_train+n_test,)
labels = np.dot(poly_features, true_w)  # y=5+1.2x-3.4x^2/2!+5.6x^3/3!
labels += np.random.normal(scale=0.1, size=labels.shape) # y=5+1.2x-3.4x^2/2!+5.6x^3/3!+зwhereз

# NumPy ndarray转换为tensor
true_w, features, poly_features, labels = [torch.tensor(x, dtype=
    torch.float32) for x in [true_w, features, poly_features, labels]]

# 训练
def evaluate_loss(net, data_iter, loss):  #@save
    """评估给定数据集上模型的损失"""
    metric = d2l.Accumulator(2)
    for X, y in data_iter:
        out = net(X)  # y^预测值
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric.add(l.sum(), l.numel())# 损失的总和,样本数量
    return metric[0] / metric[1]  # 损失/数量

def train(train_features, test_features, train_labels, test_labels, num_epochs=400): # 有关线性模型的训练函数
    loss = nn.MSELoss(reduction='none')  # 损失函数
    input_shape = train_features.shape[-1]

    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))  # 定义模型
    batch_size = min(10, train_labels.shape[0])
    train_iter = d2l.load_array((train_features, train_labels.reshape(-1,1)),
                                batch_size)  # 读取数据集作为训练和测试数据
    test_iter = d2l.load_array((test_features, test_labels.reshape(-1,1)),
                               batch_size, is_train=False)
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)  # 随机梯度下降优化
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log',
                            xlim=[1, num_epochs], ylim=[1e-3, 1e2],
                            legend=['train', 'test'])  # 绘图准备

    for epoch in range(num_epochs):
        d2l.train_epoch_ch3(net, train_iter, loss, trainer)  # 训练
        if epoch == 0 or (epoch + 1) % 20 == 0:
            animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss),
                                     evaluate_loss(net, test_iter, loss)))  # 绘图
    print('weight:', net[0].weight.data.numpy())

# 从多项式特征中选择前4个维度，用来拟合三阶多项式，即1,x,x^2/2!,x^3/3!，正常
train(poly_features[:n_train, :4], poly_features[n_train:, :4],
          labels[:n_train], labels[n_train:])
d2l.plt.show()
# 从多项式特征中选择前2个维度，即1和x，用来拟合非线性模式（如这里的三阶多项式函数）时，线性模型容易欠拟合，因为模型简单但是数据复杂
train(poly_features[:n_train, :2], poly_features[n_train:, :2],
      labels[:n_train], labels[n_train:])
d2l.plt.show()
# 从多项式特征中选择所有维度，用来拟合三阶多项式，出现过拟合。                                                                      用一个阶数过高的多项式来训练模型，在这种情况下，没有足够的数据用于学到高阶系数应该具有接近于零的值。 因此，这个过于复杂的模型会轻易受到训练数据中噪声的影响。 虽然训练损失可以有效地降低，但测试损失仍然很高。 结果表明，复杂模型对数据造成了过拟合。
train(poly_features[:n_train, :], poly_features[n_train:, :],
      labels[:n_train], labels[n_train:], num_epochs=1500)
d2l.plt.show()