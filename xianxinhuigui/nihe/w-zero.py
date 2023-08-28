import torch
from torch import nn
from d2l import torch as d2l

#根据公式y=0.05+Σ(i=1-d)0.01xi+εwhereε~N(0,0.01^2),生成一些数据
n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5  # 输入有200个，训练样本有20个使过拟合效果更明显
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05  # true_w：是一个200*1的矩阵，矩阵内容全为0.01

train_data = d2l.synthetic_data(true_w, true_b, n_train)  #用synthetic_data：合成数据集
train_iter = d2l.load_array(train_data, batch_size)# train_data：是一个包含20个样本的训练数据集，按数据集和确定批量大小读取数据存入此变量中

test_data = d2l.synthetic_data(true_w, true_b, n_test)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)  # train_data：是一个包含100个样本的测试数据集，按数据集和确定批量大小读取数据存入此变量中

# 初始化模型参数
def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)  # 初始化w,w是一个均值为0方差为1。长度为200*1的向量
    b = torch.zeros(1, requires_grad=True)  # 初始化b，是一个全0的标量
    return [w, b]

# 定义L2范数惩罚
def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2  # L2范数的实现，w中每个向量的平方和除以2

# 训练
def train(lambd):  # lambd即了λ
    w, b = init_params()  # 初始化参数
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss  #定义模型（这里使用的是线性模型），损失函数
    num_epochs, lr = 100, 0.003
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])  # 在动画中绘制数据
    for epoch in range(num_epochs):  # 开始进行100次迭代
        for X, y in train_iter:
                                                                                                                                                   # 广播机制使l2_penalty(w)成为一个长度为batch_size的向量
            l = loss(net(X), y) + lambd * l2_penalty(w)  # 损失函数中增加了L2范数惩罚项，
            l.sum().backward()  # 反向传播计算梯度
            d2l.sgd([w, b], lr, batch_size)  # 以随机梯度下降优化器来更新参数
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss))) # 绘制图像
    print('w的L2范数是：', torch.norm(w).item())

train(lambd=0)  # 忽略正则化直接训练，不用权重衰退，图像上看训练误差有了减少，但测试误差没有减少， 这意味着出现了严重的过拟合
d2l.plt.show()
train(lambd=3)  # 使用权重衰减，这里训练误差增大，但测试误差减小。 这正是我们期望从正则化中得到的效果。
d2l.plt.show()