import random
import torch
from d2l import torch as d2l

# 按照前面文本预处理文件中实现的那样，这里不再自己实现直接调用api
# 使用时光机器数据集构建词表， 并打印前10个最常用的（频率最高的）单词
tokens = d2l.tokenize(d2l.read_time_machine()) # 得到所有词元
# 因为每个文本行不一定是一个句子或一个段落，因此我们把所有文本行拼接到一起
corpus = [token for line in tokens for token in line] # 将所有词元展平，放在一维列表中
vocab = d2l.Vocab(corpus) # 根据词元建立词表
print(vocab.token_freqs[:10]) # 此表中前10高频词和他们的频次
#可以看到词频高的都是停用词，可以 被过滤，但是也有意义

# 画一个词频图，显示从高词频到低词频，词频量下降的很快,这个相当于是使用了一元语法
freqs = [freq for token, freq in vocab.token_freqs]
d2l.plot(freqs, xlabel='token: x', ylabel='frequency: n(x)',
         xscale='log', yscale='log')

# 接下来使用二元语法处理
bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])] # 将词元第一到倒数第二个，第二个到最后一个，一一匹配成为个个二元组
bigram_vocab = d2l.Vocab(bigram_tokens) # 用二元组构造词表，也就是说一个二元组对应一个索引，二元组们按照出现的词频排序
print(bigram_vocab.token_freqs[:10]) # 打印出用了二元语法的，前10个高频词组

# 接下来用三元
trigram_tokens = [triple for triple in zip(
    corpus[:-2], corpus[1:-1], corpus[2:])] # 和前面一样一一匹配，成为一个个三元组
trigram_vocab = d2l.Vocab(trigram_tokens)
print(trigram_vocab.token_freqs[:10])
# 经过显示出来的图的呈现，上面的三种方式都不适合建模

# 用马尔可夫的思想建模，那么就要从长序列中选取固定长度的序列，有两种方法
# 方法一：随机抽样，来自两个相邻的、随机的、小批量中的子序列不一定在原始序列上相邻。
def seq_data_iter_random(corpus, batch_size, num_steps):  #batch_size每个小批量中子序列样本的数目，num_steps是每个子序列中预定义的时间步数
    """使用随机抽样生成一个小批量子序列"""
    # 从corpus中随机选择一个起始位置，将corpus切片为以该位置为起点的子序列。这样做是为了使每个样本的起始位置都不同，增加数据的多样性
    corpus = corpus[random.randint(0, num_steps - 1):]# 产生随机偏移量，对长序列进行分区，随机范围包括num_steps-1。现在corpus存储随机数之后的序列部分。

    # 减去1，可以确保最后一个子序列的起始位置不会超出原始序列的边界。这样，计算得到的num_subseqs将是原始序列中可以完整形成的子序列的最大数量，而不会产生超出范围的情况
    num_subseqs = (len(corpus) - 1) // num_steps # 现在的序列长度能生成这么多个固定长度的子序列

    # 生成一个初始索引列表，用于指定每个子序列的起始位置。这些索引是等间隔的，每个索引之间相差num_steps个位置。
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))

    # 打乱初始索引的顺序，增加样本之间的随机性
    random.shuffle(initial_indices)

    def data(pos):
        # 返回从pos位置开始的长度为num_steps的子序列
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size # 计算总共的批次数量，即一个批次中能产生多少个子序列

    for i in range(0, batch_size * num_batches, batch_size): # 循环迭代批次，每次迭代增加batch_size个样本。一次生成一个批次的所有子序列
        # 在这里，initial_indices包含子序列的随机起始索引
        initial_indices_per_batch = initial_indices[i: i + batch_size] # 根据当前批次的索引范围i-i+b，获取对应的初始索引列表。得到这一个批次中所有子序列的起始索引，索引打乱过没有顺序
        X = [data(j) for j in initial_indices_per_batch] # 根据初始索引列表，获取对应的输入数据（X）。使用列表推导式，对每个初始索引调用data函数，得到一个子序列，并将它们存储在列表X中
        Y = [data(j + 1) for j in initial_indices_per_batch] # 目标序列y相当于是在对应序列x位置上的，后移一个位置的子序列
        yield torch.tensor(X), torch.tensor(Y) # 使用yield，可以实现在每次迭代中逐步生成数据，而不是一次性生成所有数据

my_seq = list(range(35))# 假设长序列为0-35
for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)# 输出生成的6个子序列对

#方法二：顺序采样
def seq_data_iter_sequential(corpus, batch_size, num_steps):  #
    """使用顺序分区生成一个小批量子序列"""

    offset = random.randint(0, num_steps)# 随机选择一个起始偏移量offset，它决定了每个批次中样本的起始位置。这样做是为了增加数据的多样性
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size # 计算总共的标记数量num_tokens，用于确定可用于构建子序列的标记数量
    Xs = torch.tensor(corpus[offset: offset + num_tokens]) # 根据起始偏移量和标记数量，获取输入数据Xs，即原始序列中对应的所有子序列
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens]) # 原始序列中对应的所有目标子序列，相比于Xs向后偏移一个位置
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1) # 将输入数据Xs和目标数据Ys进行形状变换，使其成为一个二维张量，每行表示一个样本
    num_batches = Xs.shape[1] // num_steps # 计算总共的批次数量，即一个批次中一共产生的样本个数
    for i in range(0, num_steps * num_batches, num_steps): # 循环，i=0，numstep，2numstep，。。。即每一个子序列在Xs中的起始索引
        X = Xs[:, i: i + num_steps] # 因为Xs里面是所有样本连在一起，现在一次次切割出具体的样本子序列x
        Y = Ys[:, i: i + num_steps]
        yield X, Y

for X, Y in seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y) # 打印生成的子序列

class SeqDataLoader:  #将上面的两个采样函数包装到一个类中， 以便稍后可以将其用作数据迭代器
    """加载序列数据的迭代器"""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter: # 如果输入参数表示想要使用随机采样
            self.data_iter_fn = d2l.seq_data_iter_random
        else: # 反之使用顺序采样
            self.data_iter_fn = d2l.seq_data_iter_sequential
        self.corpus, self.vocab = d2l.load_corpus_time_machine(max_tokens)#
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps) # 返回用对应的采样方式得到子序列

def load_data_time_machine(batch_size, num_steps,  use_random_iter=False, max_tokens=10000):
    """返回时光机器数据集的迭代器和词表"""
    data_iter = SeqDataLoader(
        batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab