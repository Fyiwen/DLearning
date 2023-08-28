import torch
from torch import nn
from d2l import torch as d2l

net = nn.Sequential(  # 实例化一个Sequential块并将需要的层连接在一起
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),  # 卷积层，输入通道1.输出通道6
    nn.AvgPool2d(kernel_size=2, stride=2),  # 平均池化层
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),  # 此时的输出为4维所以要经过这一层展平输入到多层感知机
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10))# 全连接层

X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)  # 输入为28*28的单通道（黑白）图片
for layer in net:  # 遍历按每一层做迭代
    X = layer(X)  #每一层的输出作为下一层的输入
    print(layer.__class__.__name__,'output shape: \t',X.shape)

# 以下进行模型训练
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)  # 读取数据集作为训练集和测试集

def evaluate_accuracy_gpu(net, data_iter, device=None):
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net, nn.Module):   # 判断两个参数大小类型是否相同
        net.eval()  # 设置为评估模式，一般在测试模型时加入
        if not device:
            device = next(iter(net.parameters())).device  # 指定和参数相同的设备
    # 正确预测的数量，总预测的数量
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:  # 每次从迭代器中拿出一个x和y
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), y.numel())  #accuracy(net(X), y)：计算所有预算正确的样本数，numel()函数：返回数组中元素的个数，在此可以求得总样本数
    return metric[0] / metric[1]  # metric[0]:分类正确的样本数，metric[1]:总的样本数


def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """用GPU训练模型"""
    global train_l, train_acc, test_acc, metric

    def init_weights(m):  # 初始化权值参数
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)  # 为每层进行初始化
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)  # 使用随机梯度下降的优化方法
    loss = nn.CrossEntropyLoss()  # 定义损失函数
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])  # 画图准备
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        metric = d2l.Accumulator(3)
        net.train()  # 一般在训练模型时加入
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)  # 预测值
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]  # 返回训练损失， metric[0]就是损失样本数目；metric[1]是训练正确的样本数；metric[2]是总的样本数
            train_acc = metric[1] / metric[2]  # 返回训练精度
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')

lr, num_epochs = 0.9, 10
train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
