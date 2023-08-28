import torch
print("1.标量和变量")
x = torch.tensor([3.0])
y = torch.tensor([2.0])
print(x+y, x*y, x/y, x**y)

x = torch.arange(4)
print("2.向量")
print(x)
print("张量中第四个元素:",x[3])
print("张量形状", x.shape)
print("张量长度", len(x))
z = torch.arange(24).reshape(2, 3, 4)
print("三维张量长度", len(z))

print("3.矩阵")
A = torch.arange(20).reshape(5, 4)
print(A)
print(A.shape)
print("矩阵最后一维的长度",A.shape[-1])
print("矩阵的转置",A.T)

print("4.矩阵的计算")
A = torch.arange(20, dtype=torch.float32).reshape(5,4)
B = A.clone()
print(A)
print(B)
print(A+B)
print(A*B)

a=2
X = torch.arange(24).reshape(2, 3, 4)
print(X)
print(a+X)
print(a*X)
print((a*X).shape)

print("5.矩阵求和运算")
print(A)
print(A.shape)
print(A.sum())  # 矩阵所有元素求和
print(A.sum(axis=0)) # 矩阵每一列分别求和， 相当于合并使得第0个维度消失
print(A.sum(axis=1))  # 矩阵每一行分别求和，相当于合并使得第一个维度消失
print(A.sum(axis=[0, 1]))  #0，1维度都去除相当于正常求和
print(A.sum(axis=1, keepdims=True))  #与前面不同的是他仍然保持原来的矩阵形状只不过合并成一列
print(A.mean())  # 矩阵所有元素的平均值
print(A.sum()/A.numel())

print("6.向量相乘点积")
x = torch.arange(4,dtype=torch.float32)
y = torch.ones(4,dtype=torch.float32)
print(x)
print(y)
print("向量点积",torch.dot(x, y))

print("7.矩阵乘向量")
print(torch.mv(A, x))

print("8.矩阵相乘")
B = torch.ones(4, 3)
print(torch.mm(A, B))

print("9.范数")
u = torch.tensor([3.0, -4.0])
print('向量的𝐿2范数:', torch.norm(u))  # 向量的𝐿2范数
print('向量的𝐿1范数:', torch.abs(u).sum())  # 向量的𝐿1范数
v = torch.ones((4, 9))
print('v:', v)
print('矩阵的𝐿2范数:', torch.norm(v))  # 矩阵的𝐿2范数

print('10.根据索引访问矩阵')
y = torch.arange(10).reshape(5, 2)
print('y:', y)
index = torch.tensor([1, 4])
print('访问矩阵第一行和第四行:', y[index])

print('11.理解pytorch中的gather()函数')
a = torch.arange(15).view(3, 5)
print('11.1二维矩阵上gather()函数')
print('a:', a)
b = torch.zeros_like(a)  # b和a形状一样但元素都是0
b[1][2] = 1  ##给指定索引的元素赋值
b[0][0] = 1  ##给指定索引的元素赋值
print('b:', b)
c = a.gather(0, b)
"""dim=0，相当于b=[ [x1,x2,x2],
[y1,y2,y2],
[z1,z2,z3] ]

如果dim=0
填入方式,下标中行被替代，列不变
[ [(x1,0),(x2,1),(x3,2)]
[(y1,0),(y2,1),(y3,2)]
[(z1,0),(z2,1),(z3,2)] ]

如果dim=1，下标中列被替代，行不变
[ [(0,x1),(0,x2),(0,x3)]
[(1,y1),(1,y2),(1,y3)]
[(2,z1),(2,z2),(2,z3)] ]
"""
print('c:', c)
d = a.gather(1, b)  # dim=1
print('d:', d)

print('11.2三维矩阵上gather()函数')
a = torch.randint(0, 30, (2, 3, 5))
print('a:', a)
index = torch.LongTensor([[[0, 1, 2, 0, 2],
                           [0, 0, 0, 0, 0],
                           [1, 1, 1, 1, 1]],
                          [[1, 2, 2, 2, 2],
                           [0, 0, 0, 0, 0],
                           [2, 2, 2, 2, 2]]])
print(a.size() == index.size())
b = torch.gather(a, 1, index)
print('b:', b)
c = torch.gather(a, 2, index)
print('c:', c)
index2 = torch.LongTensor([[[0, 1, 1, 0, 1],
                            [0, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1]],
                           [[1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [1, 1, 0, 0, 0]]])
d = torch.gather(a, 0, index2)
print('d:', d)

print('12.理解pytorch中的max()和argmax()函数')
a = torch.tensor([[1, 2, 3], [3, 3, 1]])
b = a.argmax(1)  # 压缩第1个维度，给出最大值的下标，即每一行为一组每组中最大值的列下表组成一个张量
c = a.max(1)  #第一个维度上看，最大值和最大值下标
d = a.max(1)[1]  #第一个维度上看，最大值下标
print('a:', a)
print('a.argmax(1):', b)
print('a.max(1):', c)
print('a.max(1)[1]:', d)

print('13.item()函数')
a = torch.Tensor([1, 2, 3])
print('a[0]:', a[0])  # 直接取索引返回的是tensor数据
print('a[0].item():', a[0].item())  # 获取python number