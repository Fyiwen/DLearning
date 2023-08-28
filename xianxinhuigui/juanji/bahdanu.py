import torch
from torch import nn
from d2l import torch as d2l
# 对解码器接口的代码进行了修改，新增了一个对注意力权重的异常处理
class AttentionDecoder(d2l.Decoder):
    """带有注意力机制解码器的基本接口"""
    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    @property
    def attention_weights(self):
        raise NotImplementedError

# 实现带有Bahdanau注意力的循环神经网络解码器。 首先，初始化解码器的状态，需要下面的输入：
# 编码器在所有时间步的最终层隐状态，将作为注意力的键和值；
# 上一时间步的编码器全层隐状态，将作为初始化解码器的隐状态；
# 编码器有效长度（排除在注意力池中填充词元）。
# 在每个解码时间步骤中，解码器上一个时间步的最终层隐状态将用作查询。 因此，注意力输出和输入嵌入都连结为循环神经网络解码器的输入

class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        self.attention = d2l.AdditiveAttention(
            num_hiddens, num_hiddens, num_hiddens, dropout) # 定义一个AdditiveAttention对象作为注意力机制的实现
        self.embedding = nn.Embedding(vocab_size, embed_size) # 用于对输入的one-hot向量进行嵌入操作。嵌入操作是将离散的词汇转换为连续的向量表示，可以将不同的词汇在同一个欧几里得空间上进行表示，使它们之间的距离能够反映它们的语义相似度，从而更好地被机器学习模型所理解和处理
        self.rnn = nn.GRU(
            embed_size + num_hiddens, num_hiddens, num_layers,
            dropout=dropout) # 循环层采用的是GRU，输入是当前时间步的输入嵌入向量+前一时间步的隐藏状态
        self.dense = nn.Linear(num_hiddens, vocab_size) # 一个全连接层，用于将GRU的输出转换为vocab中每个单词的概率分布

    def init_state(self, enc_outputs, enc_valid_lens, *args): # 接受Encoder的输出、有效长度和其他参数（可选），然后返回Decoder的初始状态
        # outputs的形状为(batch_size，num_steps，num_hiddens).
        # hidden_state的形状为(num_layers，batch_size，num_hiddens)
        outputs, hidden_state = enc_outputs # 分别表示Encoder的所有时间步的输出和最终的隐藏状态
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens) # 这些将作为Decoder的初始状态在后续的解码中使用

    def forward(self, X, state): # 接收Decoder的输入X和先前的隐状态state，并返回解码结果、Encoder的输出以及隐状态
        # enc_outputs的形状为(batch_size,num_steps,num_hiddens).
        # hidden_state的形状为(num_layers,batch_size,num_hiddens)
        enc_outputs, hidden_state, enc_valid_lens = state # 三个变量，分别表示Encoder的输出enc_outputs、最终隐藏状态hidden_state以及有效长度enc_valid_lens
        # 输出X的形状为(num_steps,batch_size,embed_size)
        X = self.embedding(X).permute(1, 0, 2) # 将输入序列X通过嵌入层映射到一个低维向量空间，并交换num_steps和batch_size的维度，使得X的形状变为(num_steps, batch_size, embed_size)
        outputs, self._attention_weights = [], []
        for x in X: # 逐步解码，遍历输入序列的所有时间步
            # query的形状为(batch_size,1,num_hiddens)
            query = torch.unsqueeze(hidden_state[-1], dim=1) # 首先使用解码器中上一个时间步的最后一个隐藏层的输出作为当前时间步的查询向量
            # context的形状为(batch_size,1,num_hiddens)
            context = self.attention(
                query, enc_outputs, enc_outputs, enc_valid_lens) # 利用Attention机制计算上下文向量
            # 在特征维度上连结
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1) # 将上下文向量与当前时间步的输入向量x在特征维度上拼接起来，
            # 将x变形为(1,batch_size,embed_size+num_hiddens)
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state) # 将拼接后的输入向量通过RNN模型得到当前时间步的输出out和新的隐藏状态hidden_state
            outputs.append(out) # out追加到outputs列表中
            self._attention_weights.append(self.attention.attention_weights) # 将Attention权重追加到_attention_weights列表中
        # 全连接层变换后，outputs的形状为
        # (num_steps,batch_size,vocab_size)
        outputs = self.dense(torch.cat(outputs, dim=0)) # 将所有时间步的输出张量在第0个维度（即时间步维度）上进行拼接，并通过全连接层将拼接后的张量映射到输出词表大小的维度
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state,
                                          enc_valid_lens] # 返回解码结果outputs以及更新后的Encoder的输出、最终隐状态和有效长度信息作为下一次解码的隐状态

    @property
    def attention_weights(self):
        return self._attention_weights

# 接下来，使用包含7个时间步的4个序列输入的小批量测试Bahdanau注意力解码器
encoder = d2l.Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16,num_layers=2)
encoder.eval()
decoder = Seq2SeqAttentionDecoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
decoder.eval()
X = torch.zeros((4, 7), dtype=torch.long)  # (batch_size,num_steps)
state = decoder.init_state(encoder(X), None) # 用编码器内容初始化解码器
output, state = decoder(X, state)
print(output.shape, len(state), state[0].shape, len(state[1]), state[1][0].shape)

# 训练
embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
batch_size, num_steps = 64, 10
lr, num_epochs, device = 0.005, 250, d2l.try_gpu()

train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
encoder = d2l.Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
decoder = Seq2SeqAttentionDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
net = d2l.EncoderDecoder(encoder, decoder)
d2l.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

#模型训练后，我们用它将几个英语句子翻译成法语并计算它们的BLEU分数
engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
    translation, dec_attention_weight_seq = d2l.predict_seq2seq(
        net, eng, src_vocab, tgt_vocab, num_steps, device, True) # 得到翻译结果translation和Decoder在不同时间步的注意力权重序列dec_attention_weight_seq
    print(f'{eng} => {translation}, ',
          f'bleu {d2l.bleu(translation, fra, k=2):.3f}')
    attention_weights = torch.cat([step[0][0][0] for step in dec_attention_weight_seq], 0).reshape((
    1, 1, -1, num_steps)) # 将Decoder的所有注意力权重拼接起来,第一个0：取出当前时间步的输出记录，是一个元组。该元组包含当前时间步的注意力权重张量、当前时间步的解码器输出张量和当前时间步的新隐状态张量，第二个0：取出当前时间步的注意力权重张量，是一个形状为(batch_size, num_query, num_key)的张量，第三个0：从当前注意力权重张量中选择第一个查询向量和每个键值之间的注意力权重向量，是一个形状为(batch_size, num_key)的张量
# step[0][0][0]取出了Decoder第一个时间步的第一个查询向量和Encoder的所有键值之间计算得到的注意力权重张量
# 最后的输出attention_weights是一个形状为(1, 1, fr_len, en_len)的张量，用于可视化Decoder在翻译过程中对输入句子各个位置的关注程度。
# 训练结束后，下面通过可视化注意力权重会发现，每个查询都会在键值对上分配不同的权重，这说明在每个解码步中，输入序列的不同部分被选择性地聚集在注意力池中
    d2l.show_heatmaps(
    attention_weights[:, :, :, :len(engs[-1].split()) + 1].cpu(),
    xlabel='Key positions', ylabel='Query positions')