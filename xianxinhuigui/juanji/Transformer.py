import math
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l

# 基于位置的前馈网络对序列中的所有位置的表示进行变换时使用的是同一个多层感知机（MLP）。在下面的实现中，输入X的形状（批量大小，时间步数或序列长度，隐单元数或特征维度）将被一个两层的感知机转换成形状为（批量大小，时间步数，ffn_num_outputs）的输出张量
class PositionWiseFFN(nn.Module):
    """基于位置的前馈网络，它的输入张量是一个位置向量序列"""
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
                 **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))

ffn = PositionWiseFFN(4, 4, 8) # 定义了一个前馈网络实例
ffn.eval()
print(ffn(torch.ones((2, 3, 4)))[0]) # 输入一个大小为 (2, 3, 4) 的张量（批量大小为 2，序列长度为 3，输入特征维数为 4），返回的结果是一个形状为 (2，3, 8) 的张量，输出是一个与输入序列相同长度的新序列，其中每个位置都被转换成了一个具有 8 个特征维度的向量
# 现在打印了第一个序列的输出结果，形状为（3，8），即长度为 3，每个位置都被转换成了一个具有 8 个特征维度的向量。

# 以下代码在对比不同维度的层规范化和批量规范化的效果。层规范化和批量规范化的目标相同，但层规范化是基于特征维度进行规范化。尽管批量规范化在计算机视觉中被广泛应用，但在自然语言处理任务中（输入通常是变长序列）批量规范化通常不如层规范化的效果好。
ln = nn.LayerNorm(2) # 层规范化
bn = nn.BatchNorm1d(2) # 批量规范化
X = torch.tensor([[1, 2], [2, 3]], dtype=torch.float32)
print('layer norm:', ln(X), '\nbatch norm:', bn(X))

class AddNorm(nn.Module):
    """残差连接后进行层规范化"""
    def __init__(self, normalized_shape, dropout, **kwargs): # normalized_shape的形状需要匹配xy
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X) # 首先进行残差连接，然后做层规范化
# 残差连接要求两个输入的形状相同，以便加法操作后输出张量的形状相同
add_norm = AddNorm([3, 4], 0.5)  # 规范化的形状，和xy的形状相同只是少一个维度。因为在层规范化的过程中，我们通常希望对同一批次（即第一维）和同一位置（即后面的特征维度）上的数据进行标准化，而不考虑这些数据的具体取值
add_norm.eval()
print(add_norm(torch.ones((2, 3, 4)), torch.ones((2, 3, 4))).shape)

class EncoderBlock(nn.Module):
    """Transformer编码器块"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = d2l.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout,use_bias) # 多头注意力层
        self.addnorm1 = AddNorm(norm_shape, dropout) # 第一个残差连接与层规范化模块
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens) # 位置前馈神经网络
        self.addnorm2 = AddNorm(norm_shape, dropout) # 第二个残差连接与层规范化模块

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens)) # attention 层根据输入张量 X 以及有效长度 valid_lens 计算出注意力输出，然后进行残差连接和层规范化，valid_lens 可以用于避免填充项对注意力计算的干扰
        return self.addnorm2(Y, self.ffn(Y))
# Transformer编码器中的任何层都不会改变其输入的形状
X = torch.ones((2, 100, 24))
valid_lens = torch.tensor([3, 2])
encoder_blk = EncoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5)
encoder_blk.eval()
print(encoder_blk(X, valid_lens).shape)


#下面实现的Transformer编码器的代码中，堆叠了num_layers个EncoderBlock类的实例。由于这里使用的是值范围在-1和1之间的固定位置编码，因此通过学习得到的输入的嵌入表示的值需要先乘以嵌入维度的平方根进行重新缩放，然后再与位置编码相加
class TransformerEncoder(d2l.Encoder):
    """Transformer编码器"""
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens) # 嵌入层
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout) # 位置编码
        self.blks = nn.Sequential()
        for i in range(num_layers): # 用于把编码块叠加起来变成一整个编码器
            self.blks.add_module("block"+str(i),  # 这个函数用于往nn.Sequential()序列中添加模块
                EncoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, use_bias))

    def forward(self, X, valid_lens, *args):
        # 因为位置编码值在-1和1之间，
        # 因此嵌入值乘以嵌入维度的平方根进行缩放，然后再与位置编码相加。
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens)) # 得到经过嵌入层和位置编码后的真正输入
        self.attention_weights = [None] * len(self.blks) # 用于存储每一个编码块的注意力权重
        for i, blk in enumerate(self.blks): # 遍历每一个编码器块
            X = blk(X, valid_lens) # 得到当前块的输出
            self.attention_weights[i] = blk.attention.attention.attention_weights # 存储注意力权重
        return X

# 下面指定了超参数来创建一个两层的Transformer编码器。 Transformer编码器输出的形状是（批量大小，时间步数目，num_hiddens）
encoder = TransformerEncoder(
    200, 24, 24, 24, 24, [100, 24], 24, 48, 8, 2, 0.5)
encoder.eval()
print(encoder(torch.ones((2, 100), dtype=torch.long), valid_lens).shape)

# 在DecoderBlock类中实现的每个层包含了三个子层：解码器自注意力、“编码器-解码器”注意力和基于位置的前馈网络。这些子层也都被残差连接和紧随的层规范化围绕
# 在掩蔽多头解码器自注意力层（第一个子层）中，查询、键和值都来自上一个解码器层的输出。关于序列到序列模型。在训练阶段，其输出序列的所有位置（时间步）的词元都是已知的；然而，在预测阶段，其输出序列的词元是逐个生成的。因此，在任何解码器时间步中，只有生成的词元才能用于解码器的自注意力计算中。为了在解码器中保留自回归的属性，其掩蔽自注意力设定了参数dec_valid_lens，以便任何查询都只会与解码器中所有已经生成词元的位置（即直到该查询位置为止）进行注意力计算
class DecoderBlock(nn.Module):
    """解码器中第i个块"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, i, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i # 表示当前时间步的索引
        self.attention1 = d2l.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout) # 多头注意力层
        self.addnorm1 = AddNorm(norm_shape, dropout) # 残差+层规范化
        self.attention2 = d2l.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens,
                                   num_hiddens) # 基于位置的前馈网络层
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1] # 获取编码器输出
        # state[2][self.i]表示解码器在当前时间步第i块的状态信息
        if state[2][self.i] is None: # 如果当前时间步没有前一个块的解码输出（即第一个块），则直接使用输入 X 中的信息作为键、值、查询向量进行自注意力计算
            key_values = X
        else: #
            key_values = torch.cat((state[2][self.i], X), axis=1) # 将前一个块的解码输出拼接到输入向量 X 后再作为键、值向量，以便编码器-解码器注意力机制使用
        state[2][self.i] = key_values # 更新记录当前时间步的解码器状态信息，以便在下一个时间步中使用
        if self.training: # 判断当前处于训练还是预测阶段
            batch_size, num_steps, _ = X.shape
            # dec_valid_lens的开头:(batch_size,num_steps),其中每一行是[1,2,...,num_steps]
            dec_valid_lens = torch.arange(
                1, num_steps + 1, device=X.device).repeat(batch_size, 1) # 生成在解码器中每个时间步的输出部分的有效长度 dec_valid_lens，避免模型过度关注无意义的填充部分并降低模型的泛化能力
        else:
            dec_valid_lens = None # 预测阶段，无法事先知道整个输出序列的形状和内容，因此也无法为每个时间步分配固定的有效长度，所以默认该序列中所有元素都是有效的

        # 下面这两行在做解码器中的自注意力
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        # 将解码器自注意力结果与编码器的内容，做编码器－解码器注意力。enc_outputs的开头:(batch_size,num_steps,num_hiddens)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state

# 为了便于在“编码器－解码器”注意力中进行缩放点积计算和残差连接中进行加法计算，编码器和解码器的特征维度都是num_hiddens
decoder_blk = DecoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5, 0)
decoder_blk.eval()
X = torch.ones((2, 100, 24))
state = [encoder_blk(X, valid_lens), valid_lens, [None]]
print(decoder_blk(X, state)[0].shape)

# 构建了由num_layers个DecoderBlock实例组成的完整的Transformer解码器。最后，通过一个全连接层计算所有vocab_size个可能的输出词元的预测概率值。解码器的自注意力权重和编码器解码器注意力权重都被存储下来，方便日后可视化的需要
class TransformerDecoder(d2l.AttentionDecoder):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers): # 叠加解码块
            self.blks.add_module("block"+str(i),
                DecoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, i))
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range (2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            # 存储解码器自注意力权重
            self._attention_weights[0][
                i] = blk.attention1.attention.attention_weights
            # 存储“编码器－解码器”自注意力权重
            self._attention_weights[1][
                i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights

# 依照Transformer架构来实例化编码器－解码器模型。在这里，指定Transformer的编码器和解码器都是2层，都使用4头注意力
num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10
lr, num_epochs, device = 0.005, 200, d2l.try_gpu()
ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
key_size, query_size, value_size = 32, 32, 32
norm_shape = [32]

train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps) # 读取数据集

encoder = TransformerEncoder(
    len(src_vocab), key_size, query_size, value_size, num_hiddens,
    norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
    num_layers, dropout)
decoder = TransformerDecoder(
    len(tgt_vocab), key_size, query_size, value_size, num_hiddens,
    norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
    num_layers, dropout)
net = d2l.EncoderDecoder(encoder, decoder)
d2l.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

# 训练结束后，使用Transformer模型将一些英语句子翻译成法语，并且计算它们的BLEU分数
engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
    translation, dec_attention_weight_seq = d2l.predict_seq2seq(
        net, eng, src_vocab, tgt_vocab, num_steps, device, True)
    print(f'{eng} => {translation}, ',
          f'bleu {d2l.bleu(translation, fra, k=2):.3f}')
# 当进行最后一个英语到法语的句子翻译工作时，可视化Transformer的注意力权重。编码器自注意力权重的形状为（编码器层数，注意力头数，num_steps或查询的数目，num_steps或“键－值”对的数目）
enc_attention_weights = torch.cat(net.encoder.attention_weights, 0).reshape((num_layers, num_heads,-1, num_steps)) # 将编码器多头注意力机制中产生的所有注意力权重矩阵 attention_weights 进行合并、重整形状，以便于后续可视化操作的进行
print(enc_attention_weights.shape)

# 在编码器的自注意力中，查询和键都来自相同的输入序列。因为填充词元是不携带信息的，因此通过指定输入序列的有效长度可以避免查询与使用填充词元的位置计算注意力。接下来，将逐行呈现两层多头注意力的权重。每个注意力头都根据查询、键和值的不同的表示子空间来表示不同的注意力
d2l.show_heatmaps(
    enc_attention_weights.cpu(), xlabel='Key positions',
    ylabel='Query positions', titles=['Head %d' % i for i in range(1, 5)],
    figsize=(7, 3.5))

# 为了可视化解码器的自注意力权重和“编码器－解码器”的注意力权重，我们需要完成更多的数据操作工作。例如用零填充被掩蔽住的注意力权重。值得注意的是，解码器的自注意力权重和“编码器－解码器”的注意力权重都有相同的查询：即以序列开始词元（beginning-of-sequence,BOS）打头，再与后续输出的词元共同组成序列
dec_attention_weights_2d = [head[0].tolist()
                            for step in dec_attention_weight_seq
                            for attn in step for blk in attn for head in blk] # dec_attention_weight_seq 列表中的所有矩阵按顺序展平，即将其压缩到一个一维列表中。然后，使用列表生成式对每个矩阵中的注意力头进行遍历，提取出每个头对应的权重，并将其转换为二维列表 dec_attention_weights_2d
dec_attention_weights_filled = torch.tensor(
    pd.DataFrame(dec_attention_weights_2d).fillna(0.0).values) # 由于注意力权重矩阵中可能存在缺失值（NaN），因此使用 Pandas 的 DataFrame 类型方法 fillna 将其中的缺失值替换为 0.0。然后，将得到的二维列表转换为 PyTorch 张量 dec_attention_weights_filled，并使用 reshape 函数将其重整为五维张量，形状为 (batch_size, 2, num_layers, num_heads, num_steps)
dec_attention_weights = dec_attention_weights_filled.reshape((-1, 2, num_layers, num_heads, num_steps))
dec_self_attention_weights, dec_inter_attention_weights = \
    dec_attention_weights.permute(1, 2, 3, 0, 4) # 分别表示解码器自注意力和交互注意力在不同时间步、不同块、不同头中的注意力权重矩阵
print(dec_self_attention_weights.shape, dec_inter_attention_weights.shape)

#由于解码器自注意力的自回归属性，查询不会对当前位置之后的“键－值”对进行注意力计算
d2l.show_heatmaps(
    dec_self_attention_weights[:, :, :, :len(translation.split()) + 1],
    xlabel='Key positions', ylabel='Query positions',
    titles=['Head %d' % i for i in range(1, 5)], figsize=(7, 3.5))

# 与编码器的自注意力的情况类似，通过指定输入序列的有效长度，输出序列的查询不会与输入序列中填充位置的词元进行注意力计算
d2l.show_heatmaps(
    dec_inter_attention_weights, xlabel='Key positions',
    ylabel='Query positions', titles=['Head %d' % i for i in range(1, 5)],
    figsize=(7, 3.5))









