import random

import torch

# with torch.no_grad() 则主要是用于停止autograd模块的工作，
# 以起到加速和节省显存的作用，具体行为就是停止gradient计算，从而节省了GPU算力和显存，但是并不会影响dropout和batchnorm层的行为。

# mm只能进行矩阵乘法,也就是输入的两个tensor维度只能是( n × m ) (n\times m)(n×m)和( m × p ) (m\times p)(m×p)
# bmm是两个三维张量相乘, 两个输入tensor维度是( b × n × m )和( b × m × p ), 第一维b代表batch size，输出为( b × n × p )
# matmul可以进行张量乘法, 输入可以是高维.

# python知识补充：
#  range() 函数返回的是一个可迭代对象（类型是对象），而不是列表类型， 所以打印的时候不会打印列表。
#  list() 函数是对象迭代器，可以把range()返回的可迭代对象转为一个列表，返回的变量类型为列表。
#  range(start, stop[, step])
# shuffle() 方法将序列的所有元素随机排序。shuffle()是不能直接访问的，需要导入 random 模块。举例：random.shuffle (list)
#  yield是python中的生成器


# 自己构造的数据集，所以知道实际的w和b
def create_data(w, b, nums_example):  # 构造数据集函数
    X = torch.normal(0, 1, (nums_example, len(w)))  # 随机生成均值为0，方差为1的随机数作为输入，里面有n个样本，每个样本列数是w的个数
    y = torch.matmul(X, w) + b  # y=x1w1+x2w2+...+b
    print("y_shape:", y.shape)
    y += torch.normal(0, 0.01, y.shape)  # 加入噪声# y=x1w1+x2w2+...+b+з
    return X, y.reshape(-1, 1)  # y从行向量转为列向量

true_w = torch.tensor([2, -3.4])
true_b = 4.2  # 这里给出了w和b的值，用于下面生成数据集
features, labels = create_data(true_w, true_b, 1000)  # 用这个函数生成了特征和标注，feature中每一行包含一个二维数据样本，label每一行包含一个标量
print(features[0], labels[0]) # 可以查看生成的训练样本
"""
若导入了d2l的包可以把他画成全是样本点的图
d2l.set_figsize()
d2l.plt.scatter(features[:,1].detach().numpy(),labels.detach().numpy(),1)
有一些pytorch版本里需要先用detach从计算图中取出才能转去numpy"""

# 每次读取一个小批量的样本以更新模型
def read_data(batch_size, features, lables):  #批量大小为batch-size，这是读取过程
    nums_example = len(features)
    indices = list(range(nums_example))  # 生成标号，列表形式存储
    random.shuffle(indices)  # 将序列的所有元素随机排序。这样样本是随机读取的没有特定顺序
    # 构造随机样本。把样本的顺序打乱，然后间隔相同访问，达到随机目的
    for i in range(0, nums_example, batch_size):  # range(start, stop, step)
        index_tensor = torch.tensor(indices[i: min(i + batch_size, nums_example)])
        yield features[index_tensor], lables[index_tensor]  # 通过索引访问向量，#yield就是返回一个值，并且记住这个返回的位置，下次迭代就从这个位置开始


batch_size = 10
for X, y in read_data(batch_size, features, labels):  # 读取一个小批量样本并打印
    print("X:", X, "\ny", y)
    break

##初始化模型参数，之后就用上面的样本来更新
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 每次更新需要计算损失函数关于模型参数的梯度
# 定义线性模型
def net(X, w, b):
    return torch.matmul(X, w) + b


# 定义损失函数，这里选择的是均方误差，没有做均值
def loss(y_hat, y):  # y_hat是预测值，y是真实值
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2  # y.reshape(y_hat.shape)为了加减方便，将y的形状统一为y_hat


# 定义优化算法
def sgd(params, batch_size, lr):  # 学习率lr
    with torch.no_grad():  # 主要是用于停止autograd模块的工作，使得更新的时候不参与梯度计算
        for param in params:
            param -= lr * param.grad / batch_size  ##  这里用param = param - lr * param.grad / batch_size会导致导数丢失， zero_()函数报错，这里求了均值以弥补上面损失函数没求
            param.grad.zero_()  ## 梯度设成0，导数如果丢失了，会报错‘NoneType’ object has no attribute ‘zero_’


# 训练过程
lr = 0.03
num_epochs = 3  # 把整个数据扫三遍

for epoch in range(0, num_epochs):  # 每一次对数据扫一遍
    for X, y in read_data(batch_size, features, labels):  # 每次拿出批量大小的x和y
        f = loss(net(X, w, b), y)  #把预测内容y^和y做损失计算
        # 因为`f`形状是(`batch_size`, 1)，而不是一个标量。`f`中的所有元素被加到一起，并以此计算关于[`w`, `b`]的梯度
        f.sum().backward()
        sgd([w, b], batch_size, lr)  # 使用参数的梯度更新参数
    with torch.no_grad():  # 以下内容不用算梯度,可以减少内存使用,训练集训练好模型后，在验证集这里使用with torch.no_grad()，训练集则不会计算梯度值，然后并不会改变模型的参数，只是看了训练的效果
        train_l = loss(net(features, w, b), labels)  # 计算所有样本的平均损失
        print("w {0} \nb {1} \nloss {2:f}".format(w, b, float(train_l.mean())))

print("w误差 ", true_w - w, "\nb误差 ", true_b - b)  # 比较真实与通过训练得到的参数，看成功程度