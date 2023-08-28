import torch
from torch import nn
from d2l import torch as d2l
# 以下以解决一个回归为例
'''首先生成一个数值数据集'''
n_train = 50  # 训练样本数
x_train, _ = torch.sort(torch.rand(n_train) * 5)   # 先生成再排序后的训练样本
def f(x):
    return 2 * torch.sin(x) + x**0.8 # 按照这个函数生成数据集，也就是说数据集中的数据分布满足这个函数。也就是说f是最完美的网络希望能学到的的曲线，可以拟合所有数据x
y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,))  # 计算训练样本的输出标签，这里在原函数的基础上加了噪声项
x_test = torch.arange(0, 5, 0.1)  # 生成测试样本
y_truth = f(x_test)  # 得到测试样本的真实输出标签
n_test = len(x_test)  # 测试样本数
print(n_test)

def plot_kernel_reg(y_hat): # 用于绘制所有的训练样本（样本由圆圈表示）， 不带噪声项的真实数据生成函数f（标记为“Truth”） 以及学习得到的预测函数（标记为“Pred”）。
    d2l.plot(x_test, [y_truth, y_hat], 'x', 'y', legend=['Truth', 'Pred'],
             xlim=[0, 5], ylim=[-1, 5]) # x_test测试样本，y_truth样本标签，y_hat样本预测值。绘制测试数据和预测结果的散点图
    d2l.plt.plot(x_train, y_train, 'o', alpha=0.5) # 画出由圆圈表示的所有训练样本点，alpha表示散点图透明度

# 针对这么多训练样本，首先使用平均汇聚的方法，计算所有训练样本输出的平均值，可以视为一条直线，可以看到拟合的效果很差，真实函数f（“Truth”）和预测函数（“Pred”）相差很大
y_hat = torch.repeat_interleave(y_train.mean(), n_test) # 因为拟合出来的就是一条直线，所以无论测试数据是什么，预测输出结果是唯一的。所以直接用此函数生成一个与测试集样本数量相同、每个元素都等于训练集标签的平均值的预测标签Tensory_hat
plot_kernel_reg(y_hat) # 画图


# 接下来使用非参数注意力汇聚模型来绘制预测结果，从绘制的结果会发现新的模型预测线是平滑的，并且比平均汇聚的预测更接近真实。
# X_repeat的形状:(n_test,n_train)，第i行表示第i个测试样本的特征值与训练集的所有样本的特征值配对。可以用于计算测试集样本与各个训练集样本之间的距离或相似度
X_repeat = x_test.repeat_interleave(n_train).reshape((-1, n_train)) # 首先每个测试样本的特征值都被重复了n_train次，再reshape形状
# attention_weights的形状：(n_test,n_train),每一行都包含着想得到x_test的结果y_test需要在（y_train）之间分配的注意力权重
attention_weights = nn.functional.softmax(-(X_repeat - x_train)**2 / 2, dim=1)# 计算测试集样本与每个训练集样本之间的欧氏距离平方。再softmax()函数，对样本之间的相似度（或距离）进行归一化，得到注意力权重

# y_hat的每个元素都是值的加权平均值，其中的权重是注意力权重
y_hat = torch.matmul(attention_weights, y_train) # 由注意力权重和训练输出做矩阵乘法得到测试集预测结果
plot_kernel_reg(y_hat) # 画图

# 下面观察注意力的权重，测试数据的输入相当于查询，而训练数据的输入相当于键。 因为两个输入都是经过排序的，因此由观察可知“查询-键”对越接近， 注意力汇聚得到的注意力权重就越高
d2l.show_heatmaps(attention_weights.unsqueeze(0).unsqueeze(0).detach(),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')


# 下面使用可学习参数的注意力机制

# 在注意力机制的背景中，可以使用小批量矩阵乘法来计算小批量数据中的加权平均值
X = torch.ones((2, 1, 4)) # 假设第一个小批量数据包含n个矩阵， 形状为a*b
Y = torch.ones((2, 4, 6))# 假设第二个小批量数据包含n个矩阵， 形状为b*c
print(torch.bmm(X, Y).shape) # 批量矩阵乘法输出的形状为（n，a，c）
# 下面将这个小批量矩阵乘法用于注意力机制最后的结果计算
weights = torch.ones((2, 10)) * 0.1 # 假设一个权重矩阵
values = torch.arange(20.0).reshape((2, 10)) # 假设一个值矩阵，所谓值也就是y_train的结果
torch.bmm(weights.unsqueeze(1), values.unsqueeze(-1)) # 做小批量矩阵乘法得到结果。weights的shape=(2, 1, 10)，value的shape=(2, 10, 1)最后算到(2, 1, 1)

# 定义带参数的注意力汇聚模型，其中使用小批量矩阵乘法得到预测结果， 这是按照Nadaraya-Watson核回归改造的注意力汇聚的带参数版本。除了多个w以外和上面没差别
class NWKernelRegression(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = nn.Parameter(torch.rand((1,), requires_grad=True))

    def forward(self, queries, keys, values):
        # queries和attention_weights的形状为(查询个数，“键－值”对个数)
        queries = queries.repeat_interleave(keys.shape[1]).reshape((-1, keys.shape[1])) # 每个queries的特征值都被重复，最终第i行表示第i个queries的特征值所有key的特征值配对。可以用于计算queries与各个key之间的距离或相似度
        self.attention_weights = nn.functional.softmax(
            -((queries - keys) * self.w)**2 / 2, dim=1) # 计算注意力权重

        # values的形状为(查询个数，“键－值”对个数)
        return torch.bmm(self.attention_weights.unsqueeze(1),
                         values.unsqueeze(-1)).reshape(-1) # 使用小批量矩阵乘法得到预测结果

# 训练
# 训练时需要将训练数据集变换为键和值用于训练注意力模型所以需要下面4行操作。 在带参数的注意力汇聚模型中， 任何一个训练样本的输入都会和除自己以外的所有训练样本的“键－值”对进行计算， 从而得到其对应的预测输出

X_tile = x_train.repeat((n_train, 1)) # 形状:(n_train，n_train)，每一行都包含着相同的训练输入
Y_tile = y_train.repeat((n_train, 1)) # Y_tile的形状:(n_train，n_train)，每一行都包含着相同的训练输出
# 下面的这个操作可以想象一下，相当于X_tile中每一行属于自己的信息都扔掉。比如第一行需要有除了x1_train以外的其他训练样本值作为key，第二行需要有除了x2_train以外的其他训练样本值作为key，以此类推
keys = X_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1)) # keys的形状:('n_train'，'n_train'-1)，即针对每一个训练样本，都有其他所有训练样本作为key值
values = Y_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1)) # # values的形状:('n_train'，'n_train'-1)

# 训练带参数的注意力汇聚模型时，使用平方损失函数和随机梯度下降
net = NWKernelRegression()
loss = nn.MSELoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.5)
animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])

for epoch in range(5):
    trainer.zero_grad()
    l = loss(net(x_train, keys, values), y_train) # 训练时，训练集的预测和标签算损失
    l.sum().backward()
    trainer.step()
    print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')
    animator.add(epoch + 1, float(l.sum()))

# 训练完带参数的注意力汇聚模型后可以发现： 在尝试拟合带噪声的训练数据时， 预测结果绘制的线不如之前非参数模型的平滑

keys = x_train.repeat((n_test, 1))# keys的形状:(n_test，n_train)，一行上是某一个测试样本对应的所有key值，key值对每一个测试样本都一样
values = y_train.repeat((n_test, 1)) # # value的形状:(n_test，n_train)
# 预测时，使用训练集内容作为键值对，得到测试集的预测结果
y_hat = net(x_test, keys, values).unsqueeze(1).detach()
plot_kernel_reg(y_hat)
# 为什么新的模型更不平滑了呢？ 下面看一下输出结果的绘制图： 与非参数的注意力汇聚模型相比， 带参数的模型加入可学习的参数后， 曲线在注意力权重较大的区域变得更不平滑
d2l.show_heatmaps(net.attention_weights.unsqueeze(0).unsqueeze(0).detach(),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')