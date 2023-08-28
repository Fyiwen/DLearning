import torch
from torch import nn

torch.device('cpu')  # 表示cpu
torch.device('cuda')  # 表示gpu
torch.device('cuda:1') # 表示第二台gpu

torch.cuda.device_count()  # 查询可用gpu的数量

def try_gpu(i=0):  #查询第i台gpu设备的函数
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]"""
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

try_gpu()
try_gpu(10)
try_all_gpus()


# 张量和GPU
x = torch.tensor([1, 2, 3])
print(x.device)  # 查询张量所在设备,默认是在cpu上

X = torch.ones(2, 3, device=try_gpu())  # 在第一个gpu上创建张量
print(X)

Y = torch.rand(2, 3, device=try_gpu(1))  # 在第二个GPU上创建一个随机张量
print(Y)

# 计算X+Y，最好保证他们再同一个设备上，即这里应该同在GPU且同一个GPU
Z = X.cuda(1)  # 现在XY设备不同，所以这里相当于把第一个gpu上的x复制一份到第二个GPU，以z来表示再与Y进行计算
print(X)
print(Z)
print(Y + Z)

Z.cuda(1)  # 这里是z本身就在第二个GPU上，这样调用不会返回什么不会再复制一份z,它将返回Z，而不会复制并分配新内存


# 类似地，神经网络模型可以指定设备
net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=try_gpu())  # 挪到第0号GPU去
net(X)
print(net[0].weight.data.device)


