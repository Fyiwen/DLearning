import os
import torch
from d2l import torch as d2l

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)  #一个x范围从-8到8，步长为0.1
# 激活函数relu的实现，展示其图像
y = torch.relu(x)
d2l.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))
d2l.plt.show()
# 这个函数的求导后图像展现，当输入为负时，ReLU函数的导数为0，而当输入为正时，ReLU函数的导数为1
y.sum().backward()
d2l.plot(x.detach(), x.grad, 'x', 'grad of relu', figsize=(5, 2.5))
d2l.plt.show()
# sigmoid函数实现
y = torch.sigmoid(x)
d2l.plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', figsize=(5, 2.5))
d2l.plt.show()
# 其导数，当输入为0时，sigmoid函数的导数达到最大值0.25； 而输入在任一方向上越远离0点时，导数越接近0。
x.grad.zero_()
y.sum().backward()
d2l.plot(x.detach(), x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))
d2l.plt.show()
# tanh函数
y = torch.tanh(x)
d2l.plot(x.detach(), y.detach(), 'x', 'tanh(x)', figsize=(5, 2.5))
d2l.plt.show()
# 其导数，当输入接近0时，tanh函数的导数接近最大值1。 与我们在sigmoid函数图像中看到的类似， 输入在任一方向上越远离0点，导数越接近0。
x.grad.zero_()
y.sum().backward()
d2l.plot(x.detach(), x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))
d2l.plt.show()