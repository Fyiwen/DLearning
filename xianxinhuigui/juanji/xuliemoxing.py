import torch
from torch import nn
from d2l import torch as d2l

'''生成一些数据作为样本'''
T = 1000  # 总共产生1000个点
time = torch.arange(1, T + 1, dtype=torch.float32) # 时间步1-1000
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,)) # 使用正弦函数和一些可加性噪声来生成序列数据。torch。normal返回从单独的正态分布中提取的随机数的张量，该正态分布的均值是0，标准差是0.2，提取一个形状为(T,)的张量
d2l.plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3)) # 把生成的序列数据画出来


tau = 4 # 使用当前时间步的前4个时间步的数据来预测下一个时间步的数值
features = torch.zeros((T - tau, tau)) # 初始化输入特征是个矩阵，下面赋值后每一行相当于表示要预测的xt，每一列是他对应的前面时间信息xt-tau，t-2，t-1
for i in range(tau):
    features[:, i] = x[i: T - tau + i] # 获取从i到T - tau + i的数据，并将其赋值给features的第i列
labels = x[tau:].reshape((-1, 1)) # 存储希望被预测出来的xt，形状调整成列向量。为什么直接从tau开始，是因为前面几个tau的数据，不够这么多个tau的数据来表示，这里直接舍弃，也可以采用用0补齐的方法

batch_size, n_train = 16, 600
# 只有前n_train个样本用于训练
train_iter = d2l.load_array((features[:n_train], labels[:n_train]),
                            batch_size, is_train=True)
# 初始化网络权重的函数
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
# 一个简单的多层感知机
def get_net():
    net = nn.Sequential(nn.Linear(4, 10),
                        nn.ReLU(),
                        nn.Linear(10, 1))
    net.apply(init_weights)
    return net
# 平方损失。注意：MSELoss计算平方误差时不带系数1/2
loss = nn.MSELoss(reduction='none')
def train(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.sum().backward()
            trainer.step()
        print(f'epoch {epoch + 1}, '
              f'loss: {d2l.evaluate_loss(net, train_iter, loss):f}')

net = get_net()
train(net, train_iter, loss, 5, 0.01)
# 预测
onestep_preds = net(features) # 模型预测出下一个时间步，也称单步预测
d2l.plot([time, time[tau:]],
         [x.detach().numpy(), onestep_preds.detach().numpy()], 'time',
         'x', legend=['data', '1-step preds'], xlim=[1, 1000],
         figsize=(6, 3))

# 多步预测。对于直到xt的观测序列，其在时间步t+k处的预测输出xt+k称为k步预测
multistep_preds = torch.zeros(T)
multistep_preds[: n_train + tau] = x[: n_train + tau] # 因为上面训练了600个，还有最前面的4个没参与训练的，把这些数值先存进去
for i in range(n_train + tau, T): # 利用之前的604个，以及训练好的网络得到后面三百多个时间的预测结果，存进去
    multistep_preds[i] = net(
        multistep_preds[i - tau:i].reshape((1, -1))) # 也是每次利用当前时间的前tau个预测，只不过越到后面前t个就都是预测出来的，逐渐精度下降，从下面显示的图就可以看出来

d2l.plot([time, time[tau:], time[n_train + tau:]],
         [x.detach().numpy(), onestep_preds.detach().numpy(),
          multistep_preds[n_train + tau:].detach().numpy()], 'time',
         'x', legend=['data', '1-step preds', 'multistep preds'],
         xlim=[1, 1000], figsize=(6, 3))

max_steps = 64
features = torch.zeros((T - tau - max_steps + 1, tau + max_steps)) # 只能有T - tau - max_steps + 1这么多行是因为，和前面一样，前tau个不要，这里要预测64步开外的值，所以最后的64个值也不要，列也是一样记录i-tau-max，。。。i-1

# 列i（i<tau）是来自x的观测，直接填进去，其时间步从（i）到（i+T-tau-max_steps+1）
for i in range(tau):
    features[:, i] = x[i: i + T - tau - max_steps + 1]

# 列i（i>=tau）是来自i-tau到i-1步的预测，其时间步从（i）到（i+T-tau-max_steps+1）
for i in range(tau, tau + max_steps):
    features[:, i] = net(features[:, i - tau:i]).reshape(-1)# 每一列的值都来自当前时间步i最近的tau个数据预测而来
# 也就是说多步预测，利用tau+k个最近的数据信息预测，其中tau个是已知的，k个都是经由前tau个一个个预测出来的先知道t+1，再t+2.。。。，一层层预测最终实现，只知道xt，却预测出xt+k

steps = (1, 4, 16, 64) # k步预测，k=1，4，16，64
d2l.plot([time[tau + i - 1: T - max_steps + i] for i in steps],
         [features[:, tau + i - 1].detach().numpy() for i in steps], 'time', 'x',
         legend=[f'{i}-step preds' for i in steps], xlim=[5, 1000],
         figsize=(6, 3))