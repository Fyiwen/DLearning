import torch
from torch import nn
from d2l import torch as d2l

# 照例读取训练集
batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

def get_params(vocab_size, num_hiddens, device): # 初始化循环神经网络模型的模型参数
    num_inputs = num_outputs = vocab_size # 输入输出这个参数都是词表大小。因为正式的输入和输出都是onehot编码形式，编码的长度就是词表长度。几个词就相当于有几个类别
    def normal(shape): # 用于初始化参数的函数
        return torch.randn(size=shape, device=device)*0.01
    def three(): #一次性初始化三个形状的参数
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))

    W_xz, W_hz, b_z = three()  # 更新门参数初始化
    W_xr, W_hr, b_r = three()  # 重置门参数初始化
    W_xh, W_hh, b_h = three()  # 候选隐状态参数初始化
    # 输出层参数初始化
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)

    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    for param in params: # 这几个参数都是需要计算梯度的
        param.requires_grad_(True)
    return params

def init_gru_state(batch_size, num_hiddens, device): # 用于初始化隐状态
    return (torch.zeros((batch_size, num_hiddens), device=device), )

def gru(inputs, state, params): # 完成gru中需要的计算
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z)
        R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r)
        H_tilda = torch.tanh((X @ W_xh) + ((R * H) @ W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = H @ W_hq + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,) # 返回这个其实是为了模型的一致性，因为RNN、GRU、LSTM可以由同一个模型框架定义LSTM中多一个记忆单元会输出（H，C）所以这里即便没有也给他留个位置

vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
num_epochs, lr = 500, 1
model = d2l.RNNModelScratch(len(vocab), num_hiddens, device, get_params,
                            init_gru_state, gru) # 可以看到用的就是前面RNN的那个模型框架，只是计算方式，模型参数等内容变了
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)

'''
num_inputs = vocab_size
gru_layer = nn.GRU(num_inputs, num_hiddens)
model = d2l.RNNModel(gru_layer, len(vocab))
model = model.to(device)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
'''