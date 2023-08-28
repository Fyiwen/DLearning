import torch
# 打印一个包,里面包含可参数化的概率分布和采样参数,dir函数列出对象的所有属性
print(dir(torch.distributions))
print("张量的创建")

m = torch.ones(4)
print('四个元素全为1的张量：', m)

x = torch.arange(12)  # 0-11共十二个元素
print("x:", x)
print("x的形状为:", x.shape)

y = x.reshape(3, 4)  # 将张量重新构造成3行4列的矩阵内容不变
print("y:", y)
print("张量中元素总个数：", y.numel())

z = torch.zeros(2, 3, 4)
print("三维元素全是0的张量", z)

w = torch.randn(2, 3, 4)
print("三维元素随机获得的张量", w)

q = torch.tensor([[1, 2, 3], [4, 3, 2], [7, 4, 3]])
print("直接指定内容的张量", q)

print("2.张量的运算")
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
print(x+y)
print(x-y)
print(x*y)
print(x/y)
print(x**y)
print(torch.exp(x))

X = torch.arange(12, dtype=torch.float32).reshape(3, 4)
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print("cat连接操作，在dim=0时，列不变行增加", torch.cat((X, Y), dim=0))
print("cat连接操作，在dim=1时，行不变列增加", torch.cat((X, Y), dim=1))
print(X == Y)
print(X < Y)
print("张量中所有元素的和", X.sum())

print("3.广播机制,相当于一个容错机制")
a = torch.arange(3).reshape(3, 1)
b = torch.arange(2).reshape(1, 2)
print(a)
print(b)
print(a+b)

print("4.索引和切片")
X = torch.arange(12, dtype=torch.float32).reshape(3, 4)
print(X)
print("打印最后一个元素，即矩阵最后一行元素", X[-1])
print("打印第一个和二的元素，即矩阵第一行和第二行元素", X[1:3])

X[1, 2] = 9  # 把第一行第二列元素改成---所有行列从0开始
print(X)

X[0:2, :] = 12  # 把第0行到第一行，所有列上的元素改成
print(X)

print("6.转换为其他python对象")
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
A = Y.numpy()
print(type(A), A)
B = torch.tensor(A)
print(type(B), B)

a = torch.tensor([3.5])
print(a, a.item(), float(a), int(a))


print("5.节约内存")
before = id(Y)  # id()函数提供了内存中引用对象的确切地址
Y = Y + X
print(id(Y) == before)

before = id(X)
X += Y
print(id(X) == before)  # 使用 X[:] = X + Y 或 X += Y 来减少操作的内存开销。

before = id(X)
X[:] = X + Y
print(id(X) == before)  # 使用 X[:] = X + Y 或 X += Y 来减少操作的内存开销
