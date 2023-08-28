import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from IPython import display
from torch.utils.tensorboard import SummaryWriter
'''已经选出了好的超参所以不再需要验证集，只用测试和训练集来看每轮的loss和acc'''

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

# 加载数据集
'''
def load_data_fashion_mnist(batch_size, root='./data'):
    """Download the fashion mnist dataset and then load into memory."""
    transform = transforms.ToTensor()
    mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform) # 如果数据集不在会自动下载
    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)
    if sys.platform.startswith('win'):
        num_workers = 0  # 0表示不用额外的进程来加速读取数据
    else:
        num_workers = 4
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_iter, test_iter
'''


def load_data_fashion_mnist(batch_size, root='./data'):
    """Download the fashion mnist dataset and then load into memory."""
    transform = transforms.ToTensor()
    mnist_full = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform) # 从数据集中加载得到训练数据
    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform) # 从数据集中加载得到测试数据，用于最终的模型评估
    # 将数据集的训练集部分再分割为训练集和验证集，以实现k折交叉验证（这里十折）
    #num_samples = len(mnist_full) # 暂时的训练集中所有样本个数
    #num_val_samples = num_samples // 10  # 训练集样本分成十份
    #indices = list(range(num_samples)) # 列出所有训练样本的索引
    #np.random.shuffle(indices) # 打乱所有训练集样本索引

    #val_indices = indices[:num_val_samples]  # 验证集索引，个数是原始训练集的十分之一，用于暂时的评估和选择超参
    #train_indices = indices[num_val_samples:]  # 训练集索引，个数是原始训练集的十分之九，用于训练

    #train_dataset = torch.utils.data.Subset(mnist_full, train_indices) # 用对应索引加载出训练和验证数据集
    #val_dataset = torch.utils.data.Subset(mnist_full, val_indices)

    if sys.platform.startswith('win'):
        num_workers = 0  # 0表示不用额外的进程来加速读取数据
    else:
        num_workers = 4

    train_iter = torch.utils.data.DataLoader(mnist_full, batch_size=batch_size, shuffle=True,
                                             num_workers=num_workers)
    #val_iter = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_iter,test_iter


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

'''
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
'''
'''
def evaluate_accuracy(data_iter):
    acc_sum, n = 0.0, 0
    tp, fn = 0, 0

    for X, y in data_iter:
        y_hat = net(X)
        l = cross_entropy(y_hat, y).sum()
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]

        # 计算召回率
        y_pred = net(X).argmax(dim=1)
        tp += (y_pred == y) .sum().item()
        fn += (y_pred != y) .sum().item()

    accuracy = acc_sum / n
    loss = l / n
    recall = tp / (tp + fn)  # 计算召回率

    return accuracy, loss, recall
'''
def evaluate_accuracy(data_iter):
    acc_sum, n = 0.0, 0
    tp = {}
    fn = {}

    for X, y in data_iter:
        y_hat = net(X)
        l = cross_entropy(y_hat, y).sum()
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]

        # 计算召回率
        y_pred = net(X).argmax(dim=1)
        for i in range (10) :
            tp[i] = tp.get(i, 0) + ((y_pred == i) & (y_pred == y)).sum().item()
            fn[i] = fn.get(i, 0) + ((y_pred != i) & (i == y)).sum().item()

    accuracy = acc_sum / n
    loss = l / n
    recall = {i: tp[i] / (tp[i] + fn[i]) for i in range(10)}  # 每个类别的召回率

    return accuracy, loss, recall


# 训练
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None,optimizer=None,l2_lambda=0.01):
    optimizer = torch.optim.SGD(params, lr=lr)
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
        test_acc,test_loss,test_recall = evaluate_accuracy(test_iter) # 在验证集上评估精度
        print('epoch %d, train_loss %.4f,test_loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n,test_loss, train_acc_sum / n, test_acc))
        print('各个类别的召回率如下：')
        print(test_recall)
        writer.add_scalars('loss', {'train_loss': train_l_sum / n,
                                    'test_loss': test_loss}, epoch)  # 可视化时这个变量的名字为loss，要存档内容，到时候在tensorboar可以查看图像
        writer.add_scalars('accuracy', {'train_accuracy': train_acc_sum / n,
                                        'test_accuracy': test_acc}, epoch)
'''
def train_ch3(net, train_iter, val_iter, loss, num_epochs, batch_size,
              params=None, lr=None, optimizer=None):

    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

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
        val_acc = evaluate_accuracy(val_iter) # 在验证集上评估精度
        print('epoch %d, loss %.4f, train acc %.3f, val acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, val_acc))
'''

if __name__ == '__main__':
    batch_size = 256
    train_iter,test_iter = load_data_fashion_mnist(batch_size)
    writer = SummaryWriter(
        '../statistic1')
    num_inputs = 784
    num_outputs = 10

    W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float)
    b = torch.zeros(num_outputs, dtype=torch.float)
    W.requires_grad_(requires_grad=True)
    b.requires_grad_(requires_grad=True)

    num_epochs, lr = 10, 0.001
    l2_lambda=0.01
    train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W, b], lr, l2_lambda)
    writer.close()
    # 可视化一下在测试集上的效果
    X, y = iter(test_iter).next()

    true_labels = get_fashion_mnist_labels(y.numpy())
    pred_labels = get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
    titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

    show_fashion_mnist(X[0:9], titles[0:9])

