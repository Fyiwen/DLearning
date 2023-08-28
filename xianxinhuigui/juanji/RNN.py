import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps) # 读取训练数据集以及对应词表

# 因为本来每个词元都有一个数字索引，为了神经网络更好计算，现在将每个索引映射为相互不同的单位向量即下面的独热编码
# 索引为0和2的独热向量如下所示
F.one_hot(torch.tensor([0, 2]), len(vocab))

# 每次采样的小批量数据形状是二维张量： （批量大小，时间步数）。 one_hot函数将这样一个小批量数据转换成三维张量， 张量的最后一个维度等于词表大小（len(vocab)）。
X = torch.arange(10).reshape((2, 5))
print(F.one_hot(X.T, 28).shape) # 这样转置之后再生成，最终获得的形状为 （时间步数，批量大小，词表大小）的输出。 这将使我们能够更方便地通过最外层的维度， 一步一步地更新小批量数据的隐状态。


# 初始化循环神经网络模型的模型参数.隐藏单元数num_hiddens是一个可调的超参数。
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size # 当训练语言模型时，输入和输出来自相同的词表。 因此，它们具有相同的维度，即词表的大小

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01 # 用于生成随机值

    # 隐藏层参数初始化
    W_xh = normal((num_inputs, num_hiddens)) # 因为他和x相乘，所以要和输入的形状一样
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # 附加梯度
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True) # 这里的每一个参数都需要进行梯度计算，是可更新的参数
    return params

# 此函数用于初始化隐状态。 返回是一个张量，张量全用0填充，形状为（批量大小，隐藏单元数）。
def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )

# 定义了如何在一个时间步内计算隐状态和输出。循环神经网络模型通过inputs最外层的维度实现循环， 以便逐时间步更新小批量数据的隐状态H
def rnn(inputs, state, params):# inputs的形状：(时间步数量，批量大小，词表大小)
    W_xh, W_hh, b_h, W_hq, b_q = params # 已经初始化好的参数
    H, = state # 隐藏状态
    outputs = []
    # X的形状：(批量大小，词表大小)
    for X in inputs: # 遍历每一个批次中的所有样本
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h) # 更新隐藏状态
        Y = torch.mm(H, W_hq) + b_q # 根据新的隐藏状态得到输出
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,) # 将串联的输出作为元组的第一个元素返回。最终的隐藏状态作为第二个元素返回

# 以下函数用于包装上述所有函数
class RNNModelScratch:
    """从零开始实现的循环神经网络模型"""
    def __init__(self, vocab_size, num_hiddens, device,get_params, init_state, forward_fn): # 初始化模型参数
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens # 数据集对应词表大小，隐藏层单元个数
        self.params = get_params(vocab_size, num_hiddens, device) #网络中要更新的参数
        self.init_state, self.forward_fn = init_state, forward_fn # 一个是初始化隐状态的函数，一个是计算新输出和隐状态的函数

    def __call__(self, X, state): # 给了当前输入xt和前一个隐藏状态Ht-1
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32) # 将输入转换成独热编码形式，现在是3维
        return self.forward_fn(X, state, self.params) # 返回根据xt和Ht-1计算得到的当前输出和Ht；由于这个模型有通用性改一下forward_fn这种就可以变成GRU或者LSTM,所以不同情况下也可以不这么解释，反正就是输入这些，根据前向传播得到一个结果

    def begin_state(self, batch_size, device): #初始化隐状态
        return self.init_state(batch_size, self.num_hiddens, device)

num_hiddens = 512 # 隐藏层单元
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,init_rnn_state, rnn) # 定义了一个没训练过的RNN网络

state = net.begin_state(X.shape[0], d2l.try_gpu()) # 初始化的隐状态
Y, new_state = net(X.to(d2l.try_gpu()), state) # 将上面定义的一个x输入这个网络后得到输出，和新的隐状态
print(Y.shape, len(new_state), new_state[0].shape)  # 可以看到输出形状是（时间步数*批量大小，词表大小）， 而隐状态形状保持不变，即（批量大小，隐藏单元数）

# 预测函数，用于输入prefix后生成prefix之后的num_preds个新字符
def predict_ch8(prefix, num_preds, net, vocab, device):  # 其中的prefix是一个用户提供的包含多个字符的字符串。 在循环遍历prefix中的开始字符时， 不断地将隐状态传递到下一个时间步，但是不生成任何输出。 这被称为预热（warm-up）期， 因为在此期间模型会自我更新（例如，更新隐状态）， 但不会进行预测。 预热期结束后，隐状态的值通常比刚开始的初始值更适合预测， 从而预测字符并输出它们
    state = net.begin_state(batch_size=1, device=device) # 初始化隐状态
    outputs = [vocab[prefix[0]]] # 输出初始化为输入的第一个字符的索引
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1)) # 访问输出中的最后一个元素（作为下一步的输入xt），还要reshape是因为输入的形状需要
    for y in prefix[1:]:  # 预热期。从输入字符串的第二个字符开始，作为输出。
        _, state = net(get_input(), state) # 在预热期输出的独热编码们不用留下，因为不像下一个预测循环中输出未知，预热期的输出都是已知的。只要留下每一次更新后的隐状态state
        outputs.append(vocab[y]) # outputs存每一轮输出，输出的形式为字符对应索引值
    # 预热期已经过了，现在预测num_preds步，输入上面最新得到的隐状态和最后一个已知字符
    for _ in range(num_preds):
        y, state = net(get_input(), state) # 网络中输出这一步的预测结果和新的隐状态
        outputs.append(int(y.argmax(dim=1).reshape(1))) # 因为输出的y是一个独热编码，不是期待的字符输出，所以这里先得到对应的一维索引，存在outputs中
    return ''.join([vocab.idx_to_token[i] for i in outputs]) # 这里将outputs中的每一个索引值，都通过查词表的方式映射出对应的字符，''.join用于将列表中的字符串元素连接成一个单独的字符串

# 将输入指定为time traveller，基于这个输入生成10个后续字符。 鉴于我们还没有训练网络，它会生成荒谬的预测结果。
predict_ch8('time traveller ', 10, net, vocab, d2l.try_gpu())

def grad_clipping(net, theta):  #用于解决梯度爆炸
    """裁剪梯度"""
    if isinstance(net, nn.Module): #net是否是nn.Module类的一个实例
        params = [p for p in net.parameters() if p.requires_grad] # 记录网络中所有需要计算梯度的参数
    else:
        params = net.params # 如果net不是nn.Module的实例，那么就假设net是一个自定义的网络对象，并且具有一个名为params的属性。这行代码将net.params赋值给params变量
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params)) # 求||g||，求每一个参数的梯度的平方和，再对所有参数的求和，再开根号
    if norm > theta: # 一旦梯度长度超过theta
        for param in params: # 对每一个参数执行梯度裁剪
            param.grad[:] *= theta / norm

def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """训练网络一个迭代周期"""
    state, timer = None, d2l.Timer() # 初始化
    metric = d2l.Accumulator(2)  # 一个是训练损失之和,一个是词元数量
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # 在第一次迭代或使用随机抽样时初始化state
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # state对于nn.GRU是个张量
                state.detach_() #从计算图中分离出张量state，使其不具有梯度
            else:
                # state对于nn.LSTM或对于我们从零开始实现的模型是个张量
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer): #updater是否是torch.optim.Optimizer类的一个实例
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1) # 调用梯度裁剪的函数，这个函数是上面自定义的
            updater.step() # 执行参数更新
        else:
            l.backward()
            grad_clipping(net, 1)
            # 因为已经调用了mean函数
            updater(batch_size=1) # 调用自定义的更新函数
        metric.add(l * y.numel(), y.numel())
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop() # 求出困惑度和训练一个词元平均时间

def train_ch8(net, train_iter, vocab, lr, num_epochs, device,use_random_iter=False):
    """训练模型"""
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    # 初始化
    if isinstance(net, nn.Module): # 根据网络是不是自定义的，选择优化器是用自定义的还是直接调用的
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device) # 用于预测的函数predict，本质上就是函数predict_ch8
    # 训练和预测
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter) # ppl为一个epoch中exp(平均训练损失)即困惑度，speed是训练一个词元的时间
        if (epoch + 1) % 10 == 0:# 每训练10轮，看一下输入一个字符串时网络的预测效果
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
    print(predict('time traveller')) # epoch全走完之后，看最后的网络训练效果，这里拿两个字符串尝试
    print(predict('traveller'))

num_epochs, lr = 500, 1
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu()) # 用顺序采样训练

net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
                      init_rnn_state, rnn)
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu(),
          use_random_iter=True) #用随机采样训练

'''
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
num_hiddens = 256
rnn_layer = nn.RNN(len(vocab), num_hiddens)

state = torch.zeros((1, batch_size, num_hiddens)) # 使用张量来初始化隐状态，它的形状是（隐藏层数，批量大小，隐藏单元数）
state.shape
X = torch.rand(size=(num_steps, batch_size, len(vocab)))
Y, state_new = rnn_layer(X, state)
Y.shape, state_new.shape # rnn_layer的“输出”（Y）不涉及输出层的计算： 它是指每个时间步的隐状态，这些隐状态可以用作后续输出层的输入

class RNNModel(nn.Module):
    """循环神经网络模型"""
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        # 如果RNN是双向的（之后将介绍），num_directions应该是2，否则应该是1。self.rnn.bidirectional=True就可以设置这个网络是双向RNN
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size) # 把输入变成适合网络的3维形状
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state) # xt和ht-1算出 yt和ht
        # 全连接层首先将Y的形状改为(时间步数*批量大小,隐藏单元数)
        # 它的输出形状是(时间步数*批量大小,词表大小)。
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state # 这里返回的输出还是onehot形式的

    def begin_state(self, device, batch_size=1): # 用于初始化隐状态的函数
        if not isinstance(self.rnn, nn.LSTM):
            # nn.GRU以张量作为隐状态
            return  torch.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens),
                                device=device)
        else:
            # nn.LSTM以元组作为隐状态
            return (torch.zeros((
                self.num_directions * self.rnn.num_layers,
                batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                        self.num_directions * self.rnn.num_layers,
                        batch_size, self.num_hiddens), device=device))
                        
#在训练模型之前，让我们基于一个具有随机权重的模型进行预测                       
device = d2l.try_gpu()
net = RNNModel(rnn_layer, vocab_size=len(vocab))
net = net.to(device)
d2l.predict_ch8('time traveller', 10, net, vocab, device) # 用还没训练的网络预测一下
num_epochs, lr = 500, 1
d2l.train_ch8(net, train_iter, vocab, lr, num_epochs, device)
'''






