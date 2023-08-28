import torch

print('1.自动梯度计算')
x = torch.arange(4.0, requires_grad=True)  # 1.将梯度附加到想要对其计算偏导数的变量，（只有True的参数才会参与求导，backward 可以追踪这个参数并且计算它的梯度）
print('x:', x)
print('x.grad:', x.grad)
y = 2 * torch.dot(x, x)  # y=2xTx结果是一个标量
print('y:', y)
y.backward()  # 执行它的反向传播函数，相当于开始计算y关于他的自变量的梯度，这里y的自变量只有x，所以求完后结果存在x.grad中
print('x.grad:', x.grad)  # 这里存储y有关x的梯度,并且结果是把x的具体值带入后的具体导数值即梯度公式4x，代入x=0123后的结果
print('x.grad == 4*x:', x.grad == 4 * x)


# 计算另一个函数
x.grad.zero_()  # 梯度值会累积所以这里先将x的梯度值清零
print("x:", x)
y = x.sum()  # 这一步向量变成了标量，只有标量才能使用反向传播函数，y的值是x的值的和
print('y:', y)
y.backward()
print('x.grad:', x.grad)  # 因为y相当于没有对x做出什么升级，所以x的梯度里面存的y关于x的梯度相当于x对自己求导，所以全是1

# 非标量变量的反向传播
x.grad.zero_()
print('x:', x)
y = x * x  # y=x^2
print("y",y)
y.sum().backward()  # 这一步sum，把向量变成了标量，才可以使用反向传播函数，相当于z=y.sum,z做反向传播，那么此时y.grad里面是z关于y的梯度这里因为求和的特殊性相当于没升级结果应该是1,x.grad里面是z关于x的梯度，从结果来看就是y关于x的梯度，是2x
print('x.grad:', x.grad)  # 即里面存2x代入x具体值的结果


def f(a):
    b = a * 2
    print(b.norm())
    while b.norm() < 1000:  # 求L2范数：元素平方和的平方根
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c


print('2.Python控制流的梯度计算')
a = torch.tensor(2.0)  # 初始化变量
a.requires_grad_(True)  # 1.将梯度赋给想要对其求偏导数的变量
print('a:', a)
d = f(a)  # 目标函数是上面那个复杂的流程
print('d:', d)
d.backward()  # 执行目标函数的反向传播函数
print('a.grad:', a.grad)  # 获取梯度

