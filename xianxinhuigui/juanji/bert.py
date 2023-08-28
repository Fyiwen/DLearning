import torch
from torch import nn
from d2l import torch as d2l

# 下面介绍BERT模型
def get_tokens_and_segments(tokens_a, tokens_b=None): # 将一个句子或两个句子作为输入，然后返回BERT输入序列（含特殊标识）及其相应的片段索引

    tokens = ['<cls>'] + tokens_a + ['<sep>'] # 在最初输入的第一个句子A的前后加上特殊标识符
    # 0和1分别标记片段A和B
    segments = [0] * (len(tokens_a) + 2) # 设置Ea=0，长度为句子a的长度+两个新增标识符
    if tokens_b is not None: # 如果输入中还有第二个句子b
        tokens += tokens_b + ['<sep>'] # 句子b末尾加上标识符后添加到标识完的a句子一起
        segments += [1] * (len(tokens_b) + 1) # 设置Eb=1
    return tokens, segments

class BERTEncoder(nn.Module): # 与TransformerEncoder不同，BERTEncoder使用片段嵌入和可学习的位置嵌入
    """BERT编码器"""
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f"{i}", d2l.EncoderBlock(
                key_size, query_size, value_size, num_hiddens, norm_shape,
                ffn_num_input, ffn_num_hiddens, num_heads, dropout, True))
        # 在BERT中，位置嵌入是可学习的，因此创建一个足够长的位置嵌入参数
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len,
                                                      num_hiddens)) # 这个张量可以被视为一个权重矩阵，其中第i行表示序列中第i个位置对应的嵌入向量

    def forward(self, tokens, segments, valid_lens):
        # 在以下代码段中，X的形状保持不变：（批量大小，最大序列长度，num_hiddens）
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.pos_embedding.data[:, :X.shape[1], :] # 只采用前和x等长的个数的位置编码信息
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X
# 演示效果
vocab_size, num_hiddens, ffn_num_hiddens, num_heads = 10000, 768, 1024, 4
norm_shape, ffn_num_input, num_layers, dropout = [768], 768, 2, 0.2
encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape, ffn_num_input,
                      ffn_num_hiddens, num_heads, num_layers, dropout)
tokens = torch.randint(0, vocab_size, (2, 8)) # tokens定义为长度为8的2个输入序列，元素的取值在[0, vocab_size)之间随机选择，其实就是表示两个长度为8的句子对应的单词的索引
segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1]])
encoded_X = encoder(tokens, segments, None)
print(encoded_X.shape)

# 预训练任务
class MaskLM(nn.Module):
    """BERT的掩蔽语言模型任务"""
    def __init__(self, vocab_size, num_hiddens, num_inputs=768, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential(nn.Linear(num_inputs, num_hiddens),
                                 nn.ReLU(),
                                 nn.LayerNorm(num_hiddens),
                                 nn.Linear(num_hiddens, vocab_size))

    def forward(self, X, pred_positions): # 两个输入：BERTEncoder的编码结果和用于预测的词元位置（也就是被遮蔽的位置）
        num_pred_positions = pred_positions.shape[1] # 得到需要预测的词元的个数
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = torch.arange(0, batch_size) # 该张量包含了0到batch_size-1之间的所有整数值
        # 假设batch_size=2，num_pred_positions=3
        # 那么batch_idx是np.array（[0,0,0,1,1,1]）
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions) # 将batch_idx的每个元素重复num_pred_positions次
        masked_X = X[batch_idx, pred_positions] # 先按照batch_idx中每个元素的值获取对应的样本，再根据pred_positions中的对应位置获取每个样本中需要预测的词元所在hidden_dim维上的向量表示
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.mlp(masked_X) # 得到掩码位置的预测结果
        return mlm_Y_hat # 输出预测结果

# 演示
mlm = MaskLM(vocab_size, num_hiddens)
mlm_positions = torch.tensor([[1, 5, 2], [6, 1, 5]]) # 被掩蔽的位置
mlm_Y_hat = mlm(encoded_X, mlm_positions) # 得到掩蔽位置的预测结果
print(mlm_Y_hat.shape) # 对于每个预测，结果的大小等于词表的大小

# 通过掩码下的预测词元mlm_Y_hat的真实标签mlm_Y，可以计算在BERT预训练中的遮蔽语言模型任务的交叉熵损失
mlm_Y = torch.tensor([[7, 8, 9], [10, 20, 30]]) # 掩码位置的真实标签
loss = nn.CrossEntropyLoss(reduction='none')
mlm_l = loss(mlm_Y_hat.reshape((-1, vocab_size)), mlm_Y.reshape(-1))
print(mlm_l.shape)

class NextSentencePred(nn.Module):
    """BERT的下一句预测任务"""
    def __init__(self, num_inputs, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        self.output = nn.Linear(num_inputs, 2)

    def forward(self, X): # 返回每个BERT输入序列的二分类预测
        # X的形状：(batchsize,num_hiddens)
        return self.output(X)
# 演示
encoded_X = torch.flatten(encoded_X, start_dim=1) # 从第一维开始后面展平
# NSP的输入形状:(batchsize，num_hiddens)
nsp = NextSentencePred(encoded_X.shape[-1])
nsp_Y_hat = nsp(encoded_X)
print(nsp_Y_hat.shape)

# 可以计算两个二元分类的交叉熵损失
nsp_y = torch.tensor([0, 1]) # 真实标签
nsp_l = loss(nsp_Y_hat, nsp_y)
print(nsp_l.shape)

# 整合代码
class BERTModel(nn.Module):
    """BERT模型"""
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 hid_in_features=768, mlm_in_features=768,
                 nsp_in_features=768):
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape,
                    ffn_num_input, ffn_num_hiddens, num_heads, num_layers,
                    dropout, max_len=max_len, key_size=key_size,
                    query_size=query_size, value_size=value_size)
        self.hidden = nn.Sequential(nn.Linear(hid_in_features, num_hiddens),
                                    nn.Tanh())
        self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)
        self.nsp = NextSentencePred(nsp_in_features)

    def forward(self, tokens, segments, valid_lens=None,
                pred_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_lens) # 先得到bert输入序列
        if pred_positions is not None: # 如果给出了掩蔽位置
            mlm_Y_hat = self.mlm(encoded_X, pred_positions) # 做掩蔽位置预测
        else:
            mlm_Y_hat = None
        # 0是“<cls>”标记的索引，下一句预测使用的输入是cls位置对应的encoded——x结果，选这个是因为这个位置的隐含状态向量已经包含了整个序列的信息，也就是这个位置的信息能代表整个序列
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, nsp_Y_hat # 返回编码后的BERT表示encoded_X、掩蔽语言模型预测mlm_Y_hat和下一句预测nsp_Y_hat
# 在预训练BERT时，最终的损失函数是掩蔽语言模型损失函数和下一句预测损失函数的线性组合














