import math
import os

import numpy as np
import torch
from d2l import torch as d2l
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
  # 对计算进行矢量化，从而利用线性代数库来快速计算，而不是在py中编写贵的for循环
n = 10000
a = torch.ones(n)
b = torch.ones(n)  # 两个全为1的1000维向量

# 使用for循环来做两向量的加法，得出其计算时间
c = torch.zeros(n)
timer = d2l.Timer()
for i in range(n):
    c[i] = a[i] + b[i]
print(c)
print("{0:.5f} sec".format(timer.stop()))  # 这里是一个format函数的格式，意思是后面的停止时间按照前面的格式输出，0表示这个格式是对后面第0个字符的约束

# 使用重载的+运算符来计算按元素的和，得出时间
timer.start()
d = a + b
print(d)
print("{0:.5f} sec".format(timer.stop()))
# 这两个比较后看出矢量化的必要性

# 以下是计算正态分布的过程
def normal(x, mu, sigma):  # 正态分布函数
    p = 1 / math.sqrt(2 * math.pi * sigma ** 2)
    return p * np.exp((- 0.5 / sigma ** 2) * (x - mu) ** 2)

# 可视化正态分布
x = np.arange(-7, 7, 0.01)  # 一个数组从-7到7步长为0.01，作为自变量x的输入值
print(x)
params = [(0, 1), (0, 2), (3, 1)]# 这里的三组参数，每一组将是作为正态分布的均值和方差
d2l.plot(x, [normal(x, mu, sigma) for mu, sigma in params], xlabel='x', ylabel='p(x)', figsize=(4.5, 2.5),
         legend=[f'mean {mu}, std {sigma}' for mu, sigma in params])
d2l.plt.show()  # 把图像画出来展现