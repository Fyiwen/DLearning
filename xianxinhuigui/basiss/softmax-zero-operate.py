import torch
from IPython import display
from d2l import torch as d2l
KMP_DUPLICATE_LIB_OK=True
# 读取训练集和测试集
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 初始化模型参数
#原始数据集中的每个样本都是的图像。 本节将展平每个图像28×28，把它们看作长度为784的向量。
#softmax回归中，我们的输出与类别一样多。 因为我们的数据集有10个类别，所以网络输出维度为10。 因此，权重将构成一个的矩阵784×10， 偏置将构成一个1×10的行向量
num_inputs = 784
num_outputs = 10

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

# softmax函数
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)  # 矩阵的每一行求和，重新生成新的矩阵
    return X_exp / partition  # 结果中每一行代表一个样本，行中的每个数据代表在该类别的概率,这里应用了广播机制，

X = torch.normal(0, 1, (2, 5))# 可以用这几行来查看softmax的作用，小测试，没什么用对于任何随机输入，我们将每个元素变成一个非负数。 此外，依据概率原理，每行总和为1。
X_prob = softmax(X)
X_prob, X_prob.sum(1)

# 定义模型，                                                                  注意将数据传递到模型之前，我们使用reshape函数将每张原始图像展平为向量。W是一个784*10的矩阵，W.shape就是[784,10]的列表，可以通过W.shape[0]来访问784

def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)

# 损失函数
"""
创建一个数据样本y_hat，其中包含2个样本在3个类别的预测概率， 以及它们对应的标签y。 有了y，我们知道在第一个样本中，第一类是正确的预测； 而在第二个样本中，第三类是正确的预测。 然后使用y作为y_hat中概率的索引， 我们选择第一个样本中第一个类的概率和第二个样本中第三个类的概率。
这样我们只用一行代码就能实现函数，对第零个样本，拿出y0的数据=0.1；对第一个样本，拿出y1的书籍，y1=2,就是拿出第三个数据=0.5
# 第一个参数[0,1]表示样本号，第二个参数y表示在第一个参数确定的样本中取数的序号
"""
#y = torch.tensor([0, 2])
#y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
#y_hat[[0, 1], y]

# 交叉熵损失函数
def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])

#cross_entropy(y_hat, y)

# 计算分类精度:正确预测数量与总预测数量之比，作为评价标准
"""
如果y_hat是矩阵，那么假定第二个维度存储每个类的预测分数。 我们使用argmax获得每行中最大元素的索引来获得预测类别。 然后我们将预测类别与真实y元素进行比较。 由于等式运算符“==”对数据类型很敏感， 因此我们将y_hat的数据类型转换为与y的数据类型一致。 结果是一个包含0（错）和1（对）的张量。 最后，我们求和会得到正确预测的数量
y_hat 是矩阵，假定第二个维度存储每个类的预测分数。使用 argmax 获得每行中最大元素的索引来获得预测类别
"""
def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)  #  每一行中元素最大的那个下标存在y_hat里面，最大的元素就可以算为预测分类的类别
    cmp = y_hat.type(y.dtype) == y  # #将y_hat转换为y的数据类型然后作比较，使用cmp函数存储bool类型，将cmp转化为y的数据类型再求和——得到找出来预测正确的类别数
    return float(cmp.type(y.dtype).sum())

"""
继续使用之前定义的变量y_hat和y分别作为预测的概率分布和标签。 可以看到，第一个样本的预测类别是2（该行的最大元素为0.6，索引为2），这与实际标签0不一致。 第二个样本的预测类别是2（该行的最大元素为0.5，索引为2），这与实际标签2一致。 因此，这两个样本的分类精度率为0.5
"""
#accuracy(y_hat, y) / len(y)

"""
泛化，对于任意数据迭代器data_iter可访问的数据集， 我们可以评估在任意模型net的精度
"""
def evaluate_accuracy(net, data_iter):  #@save
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):  # 判断一个对象是否是一个已知的类型，这里判断输入的net模型是否是torch.nn.Module类型
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:  # # 每次从迭代器中拿出一个x和y
            metric.add(accuracy(net(X), y), y.numel())  # net(X)：X放在net模型中进行softmax操作，accuracy(net(X), y)：再计算所有预算正确的样本数，numel()函数：返回数组中元素的个数，在此可以求得样本数
    return metric[0] / metric[1]  #   #metric[0]:分类正确的样本数，metric[1]:总的样本数
"""
类Accumulator，用于对多个变量进行累加。 在上面的evaluate_accuracy函数中， 我们在Accumulator实例中创建了2个变量， 分别用于存储正确预测的数量和预测的总数量。 当我们遍历数据集时，两者都将随着时间的推移而累加。
"""
class Accumulator:  #@save
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

"""
由于我们使用随机权重初始化net模型， 因此该模型的精度应接近于随机猜测。 例如在有10个类别情况下的精度为0.1
"""
#evaluate_accuracy(net, test_iter)

# 训练
"""
定义一个函数来训练一个迭代周期。 请注意，updater是更新模型参数的常用函数，它接受批量大小作为参数。 它可以是d2l.sgd函数，也可以是框架的内置优化函数。
"""
def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """训练模型一个迭代周期（定义见第3章）"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()  # #告诉pytorch我要计算梯度
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)  # 在此创建了一个长度为三的迭代器，用于累加信息
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()  # 自更新
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度， metric[0]就是损失样本数目；metric[1]是训练正确的样本数；metric[2]是总的样本数
    return metric[0] / metric[2], metric[1] / metric[2]

"""
一个在动画中绘制数据的实用程序类Animator， 它能够简化本书其余部分的代码。
"""
class Animator:  #@save
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)
"""
训练函数，在train_iter访问到的训练数据集上训练一个模型net。 该训练函数将会运行多个迭代周期（由num_epochs指定）。 在每个迭代周期结束时，利用test_iter访问到的测试数据集对模型进行评估。 我们将利用Animator类来可视化训练进度。
"""
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    """训练模型（定义见第3章）"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc
# 用小批量随机梯度下降来优化模型的损失函数，设置学习率为0.1。
lr = 0.1
# 优化算法
def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)
# 现在，我们训练模型10个迭代周期。 请注意，迭代周期（num_epochs）和学习率（lr）都是可调节的超参数。 通过更改它们的值，我们可以提高模型的分类精度
num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)

#现在训练已经完成，我们的模型已经准备好对图像进行分类预测。 给定一系列图像，我们将比较它们的实际标签（文本输出的第一行）和模型预测（文本输出的第二行）
def predict_ch3(net, test_iter, n=6):  #@save
    """预测标签（定义见第3章）"""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])

predict_ch3(net, test_iter)




