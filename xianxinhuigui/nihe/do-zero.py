import torch
from torch import nn
from d2l import torch as d2l

# 丢弃函数，以dropout（p=dropout）的概率丢弃张量输入X中的元素，重新缩放剩余部分：将剩余部分除以1.0-dropout
def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1  # assert：断言。表示程序只有在符合以下条件下才能正常运行
    # 在本情况中，所有元素都被丢弃。
    if dropout == 1:
        return torch.zeros_like(X)
    # 在本情况中，所有元素都被保留。
    if dropout == 0:
        return X
    # 普通情况
    mask = (torch.Tensor(X.shape).uniform_(0, 1) > dropout).float()  # 随机生成[0,1]之间的浮点数与dropout比较，会得到一个布尔张量，再转换成0或1，这样就可以实现一个随机的丢弃操作，0的位置被丢弃，1的位置保留
    return mask * X / (1.0 - dropout)  # 没有丢弃的部分会被改变，这里的*是哈达玛积，若A=(aij)和B=(bij)是两个同阶矩阵，若cij=aij×bij,则称矩阵C=(cij)为A和B的哈达玛积

# 通过下面几个例子来测试dropout_layer函数。 我们将输入X通过暂退法操作，暂退概率分别为0、0.5和1
X = torch.arange(16, dtype = torch.float32).reshape((2, 8))
print(X)
print(dropout_layer(X, 0.))  # 不变
print(dropout_layer(X, 0.5))  # 以0.5的概率随机变化
print(dropout_layer(X, 1.))  # 全为0

# 定义模型参数，定义具有两个隐藏层的多层感知机，每个隐藏层包含256个单元
num_inputs, num_outputs, num_hidden1, num_hidden2 = 784, 10, 256, 256

# 定义模型，我们可以将暂退法应用于每个隐藏层的输出（在激活函数之后）， 并且可以为每一层分别设置暂退概率： 常见的技巧是在靠近输入层的地方设置较低的暂退概率。
dropout1, dropout2 = 0.2, 0.5  # 将第一个和第二个隐藏层的暂退概率分别设置为0.2和0.5
class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hidden1, num_hidden2, is_training=True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        # 两个隐藏层，就有三个线性层
        self.lin1 = nn.Linear(num_inputs, num_hidden1)
        self.lin2 = nn.Linear(num_hidden1, num_hidden2)
        self.lin3 = nn.Linear(num_hidden2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))  # H1：第一个隐藏层的输出

        if self.training:  # 只有在训练模型时才使用dropout
            # 在第一个全连接层之后添加一个dropout层，即H1的结果经过dropout后变成一个新的H1
            H1 = dropout_layer(H1, dropout1)
        H2 = self.relu(self.lin2(H1))  # 把H1作为输入进第二个隐藏层
        if self.training:
            # 在第二个全连接层之后添加一个dropout层
            H2 = dropout_layer(H2, dropout2)
        out = self.lin3(H2)  # 把 H2 作为输入传递给输出层
        return out

net = Net(num_inputs, num_outputs, num_hidden1, num_hidden2)

# 训练
num_epochs, lr, batch_size = 10, 0.5, 256
loss = nn.CrossEntropyLoss(reduction='none') # 损失函数
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)  #读取fashion-mnist数据集作为训练和测试数据
print(len(train_iter))
trainer = torch.optim.SGD(net.parameters(), lr=lr)  # 以梯度下降优化
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)  # 进行训练
# acc 是准确度