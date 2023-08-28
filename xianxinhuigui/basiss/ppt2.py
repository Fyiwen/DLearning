import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from IPython import display
from torch.utils.tensorboard import SummaryWriter
'''标准一点的交叉验证4折'''

def use_svg_display():
    """Use svg format to display plot in jupyter"""
    display.set_matplotlib_formats('svg')

# 梯度下降
def sgd(params, lr, batch_size):
    # 为了和原书保持一致，这里除以了batch_size，但是应该是不用除的，因为一般用PyTorch计算loss时就默认已经
    # 沿batch维求了平均了。
    for param in params:
        param.data -= lr * param.grad / batch_size  # 注意这里更改param时用的param.data

# 呈现图片
def show_fashion_mnist(images, labels):
    use_svg_display()

    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()



def load_data_fashion_mnist(batch_size,i, root='./data'):
    """Download the fashion mnist dataset and then load into memory."""
    transform = transforms.ToTensor()
    mnist_full = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform) # 从数据集中加载得到训练数据
    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform) # 从数据集中加载得到测试数据，用于最终的模型评估
    # 将数据集的训练集部分再分割为训练集和验证集，以实现k折交叉验证（这里四折）
    num_samples = len(mnist_full) # 暂时的训练集中所有样本个数
    num_val_samples = num_samples // 4  # 训练集样本分成4份
    indices = list(range(num_samples)) # 列出所有训练样本的索引

    if i==0:
        val_indices = indices[:num_val_samples]  # 验证集索引，个数是原始训练集的4分之一，用于暂时的评估和选择超参
        train_indices = indices[num_val_samples:]  # 训练集索引，个数是原始训练集的4分之3，用于训练
    elif i==1:
        val_indices = indices[num_val_samples:2*num_val_samples]
        train_indices = indices[:num_val_samples]+indices[2*num_val_samples:]
    elif i==2:
        val_indices = indices[2*num_val_samples:3*num_val_samples]
        train_indices = indices[:2*num_val_samples]+indices[3*num_val_samples:]
    elif i==3:
        val_indices = indices[3*num_val_samples:]
        train_indices = indices[:3*num_val_samples]

    train_dataset = torch.utils.data.Subset(mnist_full, train_indices) # 用对应索引加载出训练和验证数据集
    val_dataset = torch.utils.data.Subset(mnist_full, val_indices)

    if sys.platform.startswith('win'):
        num_workers = 0  # 0表示不用额外的进程来加速读取数据
    else:
        num_workers = 4

    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                             num_workers=num_workers)
    val_iter = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_iter, val_iter


def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

# softmax函数
def softmax(X):
    X_exp = X.exp()
    # 同一列（dim=0）或同一行（dim=1）
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制


# 交叉熵损失
def cross_entropy(y_hat, y):
    return - torch.log(y_hat.gather(dim=1, index=y.view(-1, 1)))
    # return - torch.log(torch.gather(y_hat, dim=1, index=y))

# 准确度
def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()

# 网络
def net(X):
    return softmax(torch.mm(X.view((-1, num_inputs)), W) + b)

#
def evaluate_accuracy(data_iter): # 给验证集或者测试集用
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        # print(X.shape)
        # print(y.shape)
        y_hat = net(X)
        l = cross_entropy(y_hat, y).sum()
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n, l / n



# 训练
def train_ch3(net, train_iter, val_iter, loss, num_epochs, batch_size,
              params=None, lr=None,optimizer=None,l2_lambda=0.01):
    optimizer = torch.optim.SGD(params, lr=lr)
    tacc=0
    vacc=0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            # 添加L2正则化项
            if params is not None:
                l2_regularization = torch.tensor(0.)
                for param in params:
                    l2_regularization += torch.norm(param, p=2)  # 计算权重参数的L2范数
                l += l2_lambda * l2_regularization

            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            if optimizer is None:
                sgd(params, lr, batch_size)
            else:
                optimizer.step()  # “softmax回归的简洁实现”一节将用到

            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        val_acc,val_loss = evaluate_accuracy(val_iter) # 在验证集上评估精度
        print('epoch %d, train_loss %.4f,val_loss %.4f, train acc %.3f, val acc %.3f'
              % (epoch + 1, train_l_sum / n,val_loss, train_acc_sum / n, val_acc))
        writer.add_scalars('loss', {'train_loss': train_l_sum / n,
                                    'val_loss': val_loss},epoch)  # 可视化时这个变量的名字为loss，要存档内容，到时候在tensorboar可以查看图像
        writer.add_scalars('accuracy', {'train_accuracy': train_acc_sum / n,
                                        'val_accuracy': val_acc}, epoch)
        tacc=train_acc_sum / n # 记录每一轮的最终训练精度，而且永远只记得最新一轮的结果
        vacc=val_acc
    return tacc, vacc
if __name__ == '__main__':
    batch_size = 256
    writer = SummaryWriter(
        '../statistic')  # SummaryWriter的作用就是，将数据以特定的格式存储到这个路径的文件夹中。首先我们实例化writer之后我们使用这个writer对象“拿出来”的任何数据都保存在这个路径之下，之后tensorboard可以可视化这些数据
    acc1=0
    acc2=0
    for i in range(4):
        train_iter, val_iter = load_data_fashion_mnist(batch_size,i)

        num_inputs = 784 # 28*28的图像
        num_outputs = 10

        W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float)
        b = torch.zeros(num_outputs, dtype=torch.float)
        W.requires_grad_(requires_grad=True)
        b.requires_grad_(requires_grad=True)

        num_epochs, lr = 5, 0.001
        l2_lambda=0.01

        tacc,vacc=train_ch3(net, train_iter, val_iter, cross_entropy, num_epochs, batch_size, [W, b], lr, l2_lambda)
        acc1+=tacc # 每一组交叉验证中得到的训练精度
        acc2+=vacc # 每一组交叉验证中得到的验证精度
    print('train_acc %.4f,val_acc %.4f'
              % (acc1 / 4, acc2/4 ))
    writer.close()


