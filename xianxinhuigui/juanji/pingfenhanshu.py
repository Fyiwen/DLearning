import math
import torch
from torch import nn
from d2l import torch as d2l

# 实现掩蔽softmax操作，其中任何超出有效长度的位置都被掩蔽并置为0
def masked_softmax(X, valid_lens):
    """通过在最后一个轴上即隐藏单元的维度，掩蔽元素来执行softmax操作"""
    # X是一个形状为(batch_size, seq_len, num_hiddens)的三维张量
    # valid_lens为一个与X的第一维大小相同的一维张量，表示每个序列的有效长度
    if valid_lens is None: # 在没有valid_lens参数时，该函数直接调用函数来进行标准的softmax运算,不遮蔽
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1: # 如果valid_lens的维度为1，则其表示每个序列的有效长度，形状为（batch_size）
            valid_lens = torch.repeat_interleave(valid_lens, shape[1]) # 将valid_lens中的每个值都重复shape[1]次,也就是变成一个(batch_size * seq_len)的一维张量
        else: # 否则，需要将其重塑为形状为(batch_size * seq_len)的一维张量
            valid_lens = valid_lens.reshape(-1)

        X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                              value=-1e6)# 该函数根据valid_lens参数产生一个掩码矩阵mask，形状与输入二维张量相同。其中每行的前valid_lens[i]个元素为1，表示该位置是有效的；而剩余的元素为0，表示该位置是填充符号。然后将mask中为0的位置赋值为一个很小的负数，以便在后面的softmax计算中忽略它们的影响。
        return nn.functional.softmax(X.reshape(shape), dim=-1) #最后，在计算完带遮蔽的softmax之后，需要将结果重塑为与X相同的形状，即(batch_size, seq_len, num_hiddens)

# 演示此函数是如何工作的， 考虑由两个2*4矩阵表示的样本， 这两个样本的有效长度分别为2和3.经过掩蔽softmax操作，超出有效长度的值都被掩蔽为0
masked_softmax(torch.rand(2, 2, 4), torch.tensor([2, 3]))
# 同样也可以使用二维张量，为矩阵样本中的每一行指定有效长度
masked_softmax(torch.rand(2, 2, 4), torch.tensor([[1, 3], [2, 4]]))

# 加性注意力实现
# 将查询和键连结起来后输入到一个多层感知机（MLP）中， 感知机包含一个隐藏层，其隐藏单元数是一个超参数h，通过使用tanh作为激活函数，并且禁用偏置项
class AdditiveAttention(nn.Module):
    """加性注意力"""
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys) # 分别进行上面的线性变换，将queries和keys转换到特征空间
        # queries的形状：(batch_size，查询的个数，1，num_hidden)
        # key的形状：(batch_size，1，“键－值”对的个数，num_hiddens)
        features = queries.unsqueeze(2) + keys.unsqueeze(1) # # 使用广播机制，将queries和keys融合得到一个特征张量
        features = torch.tanh(features) # # 输入tanh函数，确保值域在[-1, 1]之间
        # self.w_v仅有一个输出，因此从形状中移除最后那个维度。
        # scores的形状：(batch_size，查询的个数，“键-值”对的个数)
        scores = self.w_v(features).squeeze(-1) # 投影特征向量到一个标量得分
        self.attention_weights = masked_softmax(scores, valid_lens) #  # 对得分进行遮蔽softmax操作，获得注意力权重
        # values的形状：(batch_size，“键－值”对的个数，值的维度)
        return torch.bmm(self.dropout(self.attention_weights), values) # 将得到的注意力权重与对应的values向量融合得到最终结果

# 用一个小例子来演示其中查询、键和值的形状为（批量大小，步数或词元序列长度，特征大小）， 实际输出为
# (2,1,20),(2,10,2)、(2,10,4).注意力汇聚输出的形状为（批量大小，查询的步数，值的维度）
queries, keys = torch.normal(0, 1, (2, 1, 20)), torch.ones((2, 10, 2))
# values的小批量，两个值矩阵是相同的
values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(2, 1, 1)
valid_lens = torch.tensor([2, 6])
attention = AdditiveAttention(key_size=2, query_size=20, num_hiddens=8,
                              dropout=0.1) # 使用加性注意力的模型对象
attention.eval()
attention(queries, keys, values, valid_lens) # 用上面定义的实例看使用加性计算后的结果
# 尽管加性注意力包含了可学习的参数，但由于本例子中每个键都是相同的， 所以注意力权重是均匀的，由指定的有效长度决定
d2l.show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries')

# 下面的缩放点积注意力的实现使用了丢弃法进行模型正则化
class DotProductAttention(nn.Module):
    """缩放点积注意力"""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # queries的形状：(batch_size，查询的个数，d)
    # keys的形状：(batch_size，“键－值”对的个数，d)
    # values的形状：(batch_size，“键－值”对的个数，值的维度)
    # valid_lens的形状:(batch_size，)或者(batch_size，查询的个数)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1] # 获取查询向量queries的最后一个维度，表示embedding的维度
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d) # 使用点积计算得到scores，并除以sqrt(d)；结果是一个(batch_size, seq_length_q, seq_length_k)的矩阵
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)  # 将得到的注意力权重与对应的values向量融合得到最终结果

# 为了演示,使用与先前加性注意力例子中相同的键、值和有效长度。 对于点积操作，我们令查询的特征维度与键的特征维度大小相同
queries = torch.normal(0, 1, (2, 1, 2))
attention = DotProductAttention(dropout=0.5)
attention.eval()
attention(queries, keys, values, valid_lens)
# 与加性注意力演示相同，由于键包含的是相同的元素， 而这些元素无法通过任何查询进行区分，因此获得了均匀的注意力权重
d2l.show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries')

