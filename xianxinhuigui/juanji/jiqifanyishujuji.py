import os
import torch
from d2l import torch as d2l
# 下载一个英-法数据集，数据集中的每一行都是制表符分隔的文本序列对， 序列对由英文文本序列和翻译后的法语文本序列组成
d2l.DATA_HUB['fra-eng'] = (d2l.DATA_URL + 'fra-eng.zip',
                           '94646ad1522d915e7b0f9296181140edcf86a4f5')
#
def read_data_nmt():
    """载入“英语－法语”数据集"""
    data_dir = d2l.download_extract('fra-eng') # 数据集所在路径
    with open(os.path.join(data_dir, 'fra.txt'), 'r',
             encoding='utf-8') as f:
        return f.read() # 读取整个文件内容

raw_text = read_data_nmt() # 读取数据集内容
print(raw_text[:75]) # 显示读取的前75个字符

def preprocess_nmt(text):
    """预处理“英语－法语”数据集"""
    def no_space(char, prev_char): # 输入当前字符和前一个字符
        return char in set(',.!?') and prev_char != ' ' # 判断给定的字符是否是逗号、句号、问号或感叹号，并且前一个字符不是空格

    # 使用空格替换特殊字符'\u202f'和'\xa0'。这些特殊字符通常是空格字符的变体
    # 使用小写字母替换大写字母
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # 在单词和标点符号之间插入空格
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char # 如果当前char不是第一个单词，而且是前面没空格的标点符号，则在他前面加个空格
           for i, char in enumerate(text)] # 遍历每一个字符
    return ''.join(out) # 将列表中的元素以空字符串作为分隔符进行连接

text = preprocess_nmt(raw_text) # 对读取的数据集内容进行预处理
print(text[:80])

def tokenize_nmt(text, num_examples=None):
    """词元化“英语－法语”数据数据集""" # 在以前的代码中为了简便，使用的是字符词元化，这里比较适合单词词元化
    source, target = [], []
    for i, line in enumerate(text.split('\n')): # 将文本根据换行符('\n')进行分割，遍历预处理后的数据集文本每一行，即每一对序列
        if num_examples and i > num_examples: # 这个num_examples可以用于限定选择数据集中的几行内容进行词元化，一旦达到指定的示例数量就可以结束对文本的处理
            break
        parts = line.split('\t') # 对当前行文本进行按制表符('\t')进行分割，因为这个数据集中就是用制表符分隔对应的英语-法语句子
        if len(parts) == 2: # 判断列表长度是否为2，以确保当前行包含了两个部分即英语和法语数据
            source.append(parts[0].split(' ')) # 当前英语句子中，按照空格分出一个个单词或符号，作为词元
            target.append(parts[1].split(' ')) # 法语句子中
    return source, target # source[i]是源语言（这里是英语）第i个文本序列的词元列表，词元列表中一个词元可能是一个单词或一个符号。同理target目标语言

source, target = tokenize_nmt(text) # 对数据集内容词元化
print(source[:6], target[:6]) # 分别展示前6个词元

# 绘制每个文本序列所包含的词元数量的直方图
def show_list_len_pair_hist(legend, xlabel, ylabel, xlist, ylist):
    """绘制列表长度对的直方图"""
    d2l.set_figsize()
    _, _, patches = d2l.plt.hist(
        [[len(l) for l in xlist], [len(l) for l in ylist]]) # 这两个列表分别表示xlist和ylist中各元素的长度。就是每一个序列中词元个数hist()函数将根据这些长度值绘制直方图，并返回三个值，但这里代码只接收并使用了最后一个值patches，是一个Bar容器对象，该对象包含了绘制的直方图柱子的各种属性和样式
    d2l.plt.xlabel(xlabel)
    d2l.plt.ylabel(ylabel)
    for patch in patches[1].patches: # patches[1]表示Bar容器对象中的第二个元素，它可能包含了多个直方图柱子的信息，
        patch.set_hatch('/') # 柱子的填充样式为斜线
    d2l.plt.legend(legend) # 图例
show_list_len_pair_hist(['source', 'target'], '# tokens per sequence',
                        'count', source, target)

# 为源语言构建两个词表，构建词表的方法和前面文本预处理的一样，这里直接调用
# 使用单词级词元化时，词表大小将明显大于使用字符级词元化时的词表大小。 为了缓解这一问题，这里将出现次数少于2次的低频率词元视为相同的未知（“<unk>”）词元。 除此之外，还指定了额外的特定词元， 例如在小批量时用于将序列填充到相同长度的填充词元（“<pad>”）， 以及序列的开始词元（“<bos>”）和结束词元（“<eos>”）。 这些特殊词元在自然语言处理任务中比较常用。
src_vocab = d2l.Vocab(source, min_freq=2,
                      reserved_tokens=['<pad>', '<bos>', '<eos>'])
len(src_vocab)

# 由于机器翻译中输入的序列不等长，为了简化处理还是得统一成固定长
def truncate_pad(line, num_steps, padding_token): # 输入line是一个序列的对应在词表中的索引列表
    """截断或填充文本序列，使得不等长的序列变成固定长 num_steps，"""
    if len(line) > num_steps: # 序列大于固定长
        return line[:num_steps]  # 截断，只取其前num_steps 个词元
    return line + [padding_token] * (num_steps - len(line))  # 文本序列的索引数目也就是词元数目少于num_steps时， 我们将在其末尾添加特定的“<pad>”词对应的索引值

truncate_pad(src_vocab[source[0]], 10, src_vocab['<pad>'])

def build_array_nmt(lines, vocab, num_steps): # lines是词元列表source或者target
    """将机器翻译的文本序列转换成小批量"""
    lines = [vocab[l] for l in lines] # 遍历词元列表中每一个文本序列对应的词元列表，得到这些词元在词表中对应索引，所以现在这是索引列表。
    lines = [l + [vocab['<eos>']] for l in lines] # 将特定的“<eos>”词元对应的索引添加到所有序列的末尾， 用于表示序列的结束。当模型通过一个词元接一个词元地生成序列进行预测时， 生成的“<eos>”词元说明完成了序列输出工作
    array = torch.tensor([truncate_pad(
        l, num_steps, vocab['<pad>']) for l in lines]) # array是经过填充截断处理后的lines
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1) # 生成一个布尔型的张量，其中元素为True表示对应位置上的词汇不是填充词汇，为False表示是填充词汇。然后，使用.type(torch.int32)将布尔型张量转换为整型张量，并使用.sum(1)计算每个句子中非填充词汇的数量。这样，valid_len就得到了每个句子的有效长度
    return array, valid_len # 固定长索引序列，序列的有效长度

# 综合操作如下
def load_data_nmt(batch_size, num_steps, num_examples=600):
    """返回翻译数据集的迭代器和词表"""
    text = preprocess_nmt(read_data_nmt()) # 数据集读取+预处理
    source, target = tokenize_nmt(text, num_examples) # 词元化数据集，得两个词元列表，一个源语言一个目标语言。source[i]是源语言第i个文本序列的词元列表
    src_vocab = d2l.Vocab(source, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>']) # 源语言词表
    tgt_vocab = d2l.Vocab(target, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>']) # 目标语言词表
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps) # 所有源语言序列的索引序列，序列有效长度们
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = d2l.load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab

# 数据集中的第一个小批量数据读出来看看
train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=2, num_steps=8)
for X, X_valid_len, Y, Y_valid_len in train_iter:
    print('X:', X.type(torch.int32)) # 会发现有两个序列，值是词表索引，因为batch=2。打印出X的值。在打印之前，使用X.type(torch.int32)将X的数据类型转换为torch.int32，确保输出的数据类型是整型
    print('X的有效长度:', X_valid_len) # 两个有效长度
    print('Y:', Y.type(torch.int32))
    print('Y的有效长度:', Y_valid_len)
    break