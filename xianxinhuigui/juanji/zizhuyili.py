import math
import torch
from torch import nn
from d2l import torch as d2l

num_hiddens, num_heads = 100, 5 # num_heads注意力头的数量
attention = d2l.MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,num_hiddens, num_heads, 0.5) # 创建了一个MultiHeadAttention实例，第一个num_hiddens表示查询、键、值向量的维度；第二个num_hiddens表示解码器的隐藏状态的维度；第三个num_hiddens表示输出向量的维度；第四个num_hiddens表示残差连接的维度；num_heads表示注意力头的数量；最后一个参数0.5是dropout的概率
attention.eval()

batch_size, num_queries, valid_lens = 2, 4, torch.tensor([3, 2]) # valid_lens是一个长度为2的张量，指示了两个输入句子的有效标记数
X = torch.ones((batch_size, num_queries, num_hiddens)) # 定义了一个形状为(2, 4, 100)的张量X
print(attention(X, X, X, valid_lens).shape) # X作为查询、键、值张量的输入注意力机制，并且得到输出

class PositionalEncoding(nn.Module):
    """位置编码，用于对序列中的每个位置嵌入可区分的信息"""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建一个足够长的P，用于存储位置编码
        self.P = torch.zeros((1, max_len, num_hiddens))
        # 首先创建了一个形状为 (max_len,1) 的张量 X，表示每个位置。然后计算出一个与 X 相同形状的位置编码表，并将其存入 P 张量中
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X) # 偶数位置的编码信息
        self.P[:, :, 1::2] = torch.cos(X) # 奇数位置上的编码信息

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device) # 将x与对应位置的位置编码相加
        return self.dropout(X)

# 在位置嵌入矩阵P中， 行代表词元在序列中的位置，列代表位置编码的不同维度。 从下面的例子中可以看到位置嵌入矩阵的第6列和第7
# 列的频率高于第8列和第9列。 第6列和第7列之间的偏移量（第 8列和第9列相同）是由于正弦函数和余弦函数的交替
encoding_dim, num_steps = 32, 60
pos_encoding = PositionalEncoding(encoding_dim, 0)
pos_encoding.eval()
X = pos_encoding(torch.zeros((1, num_steps, encoding_dim))) # 将 X 作为输入传入位置编码器中，计算出位置编码张量 P，输出结合后的结果
P = pos_encoding.P[:, :X.shape[1], :] # 由于位置编码器的最大序列长度是 1000，而当前的输入序列长度是 num_steps，因此我们只需要截取前 num_steps 个位置对应的编码向量即可
d2l.plot(torch.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
         figsize=(6, 2.5), legend=["Col %d" % d for d in torch.arange(6, 10)]) # 其横坐标表示输入序列中的位置，纵坐标表示每个位置嵌入向量的各维度，P[0, :, 6:10].T 表示选取 P 张量中第一个样本，而且只选取第 6~9 列进行可视化。在图像中，每列颜色不同，代表了不同的编码信息。

# 打印出0-8的二进制表示形式
for i in range(8):
    print(f'{i}的二进制是：{i:>03b}')
#在二进制表示中，较高比特位的交替频率低于较低比特位， 与下面的热图所示相似，只是位置编码通过使用三角函数在编码维度上降低频率。 由于输出是浮点数，因此此类连续表示比二进制表示法更节省空间

P = P[0, :, :].unsqueeze(0).unsqueeze(0) # 第一个样本的所有位置信息，改变形状后适应下面的函数
d2l.show_heatmaps(P, xlabel='Column (encoding dimension)',
                  ylabel='Row (position)', figsize=(3.5, 4), cmap='Blues')
# 每一行对应输入序列中的一个位置，每一列对应位置编码向量的一个维度，颜色越浅表示该维度值越大







