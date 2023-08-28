import collections
import re
from d2l import torch as d2l
# 下载了一本书，作为这里的数据集
d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')

def read_time_machine():  #
    """将时间机器数据集加载到文本行的列表中"""
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines() # 将这个数据集按行读取，存入lines
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines] # 处理每一行把除了字母以外的所有内容变成空格，去除字符串两端的空白字符，字母都变成小写

lines = read_time_machine() # 得到处理后的内容
print(f'# 文本总行数: {len(lines)}')
print(lines[0])
print(lines[10])

def tokenize(lines, token='word'):  #将文本行列表（lines）作为输入， 列表中的每个元素是一个文本序列，每个文本序列又被拆分成一个词元列表，词元（token）是文本的基本单位。 最后，返回一个由词元列表组成的列表，其中的每个词元都是一个字符串（string）
    """将文本行拆分为单词或字符词元"""
    if token == 'word': #如果参数时word表示需要将每一行中拆分成单词词元
        return [line.split() for line in lines]
    elif token == 'char': #如果参数时char表示需要将每一行中拆分成字符词元
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)

tokens = tokenize(lines) # 得到所有词元
for i in range(11):
    print(tokens[i])

class Vocab:  #词元的类型是字符串或者字符，而模型需要的输入是数字，因此构建一个字典，通常也叫做词表（vocabulary）， 用来将词元映射到从0开始的数字索引中。
    """文本词表"""# 然后根据每个唯一词元的出现频率，为其分配一个数字索引。另外也可以选择增加一个列表，用于保存那些被保留的词元， 例如：填充词元（“<pad>”）； 序列开始词元（“<bos>”）； 序列结束词元（“<eos>”）
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = [] # 里面存预留的特殊词汇，默认为空，如果不空可以作为输入参数给

        counter = count_corpus(tokens)# 先将训练集中的所有词元合并在一起，对它们的唯一词元进行统计，得到的统计结果称之为语料（corpus）
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True) # 按照出现频次从高到低进行排序，存储排序后的词汇及其频次

        self.idx_to_token = ['<unk>'] + reserved_tokens # 这个用于存储下面这个字典中已经记录的词汇。这一步在初始化，默认初始时就记录了未知词汇和预留的特殊词汇，下面会有他们对应索引
        self.token_to_idx = {token: idx # 这一步也是初始化，按照下面规则<unk>对应索引0，reserved_tokens中对应索引1，2.。。依次往后
                             for idx, token in enumerate(self.idx_to_token)} # 这是是一个字典，用于查找词汇对应的索引
        for token, freq in self._token_freqs: # 遍历每一个词汇
            if freq < min_freq: # 低于最小频次的词元移除，不参与词表构建
                break
            if token not in self.token_to_idx: # 如果这个词汇不在现有字典中
                self.idx_to_token.append(token) # 把这个词汇加入，这样idx_to_token中存储所有已经在字典中记录的词汇
                self.token_to_idx[token] = len(self.idx_to_token) - 1 # 这个词汇对应的索引存在字典中方便后面查询，索引值就是这个len-1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):# 根据输入的词汇或词汇列表，返回对应的索引或索引列表
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):# 根据输入的索引或索引列表，返回对应的词汇或词汇列表
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # 返回未知词元的索引为0。语料库中不存在或已删除的任何词元都将映射到一个特定的未知词元“<unk>”
        return 0

    @property
    def token_freqs(self): # 回词汇表中每个词汇及其对应的频次列表
        return self._token_freqs

def count_corpus(tokens):  #输入所有的词元
    """统计词元的频率"""
    # tokens可能1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list): # 长度是否为0，或者tokens的第一个元素是否为列表？满足条件的是2D，需要展平成1维
        tokens = [token for line in tokens for token in line] # 外层循环 for line in tokens循环遍历每一行，内层循环 for token in line循环遍历当前行中的词元
    return collections.Counter(tokens) # 遍历tokens中的元素，并计算每个元素的频次。最终，函数返回一个包含词汇频次统计结果的计数器对象。

vocab = Vocab(tokens) # 构建数据集对应词表，
print(list(vocab.token_to_idx.items())[:10])#打印前几个高频词元及其索引

for i in [0, 10]:
    print('文本:', tokens[i]) # 打印文本行
    print('索引:', vocab[tokens[i]]) # 将每一条文本行转换成一个数字索引列表




# 以上都是细节，下面是一个上述总体功能的调用
def load_corpus_time_machine(max_tokens=-1): # max_tokens用于限制词元索引列表的长度
    """返回时光机器数据集的词元索引列表和词表"""
    lines = read_time_machine()
    tokens = tokenize(lines, 'char') #按照字符，词元化文本行
    vocab = Vocab(tokens) #建立词表
    # 因为时光机器数据集中的每个文本行不一定是一个句子或一个段落，可能是一个单词所以将所有文本行展平到一个列表中
    corpus = [vocab[token] for line in tokens for token in line] # 得到每一个词元的索引
    if max_tokens > 0: # 如果传入的max_tokens大于0，则将corpus列表进行切片操作，保留前max_tokens个词元索引。这一步限制了corpus的长度，使其最多包含指定数量的词元索引
        corpus = corpus[:max_tokens] #一旦词元索引列表的长度超过，就切片
    return corpus, vocab

corpus, vocab = load_corpus_time_machine()
len(corpus), len(vocab)