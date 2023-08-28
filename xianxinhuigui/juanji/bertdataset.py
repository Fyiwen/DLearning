import os
import random
import torch
from d2l import torch as d2l

def _read_wiki(data_dir):
    file_name = os.path.join(data_dir, 'wiki.train.tokens')
    with open(file_name, 'r') as f:
        lines = f.readlines() # 读取所有行的内容
    # 大写字母转换为小写字母
    paragraphs = [line.strip().lower().split(' . ')
                  for line in lines if len(line.split(' . ')) >= 2] # 先按照句号进行切割，去除每个句子开头和结尾的空白符，并将所有字母转换为小写字母，过滤掉长度小于2的无效句子
    random.shuffle(paragraphs)
    return paragraphs

# 首先为BERT的两个预训练任务实现辅助函数
def _get_next_sentence(sentence, next_sentence, paragraphs): # 函数生成二分类任务的训练样本,即生成下一句预测任务的数据
    if random.random() < 0.5: # 随机数小于0.5就使用下一句作为“下一句预测”任务的正样本
        is_next = True
    else:
        next_sentence = random.choice(random.choice(paragraphs)) # 否则，从paragraphs中随机选择一段文本，并从该文本中随机选择一句话作为下一句（负样本），并将is_next变量设置为False，表示使用该负样本作为“下一句预测”任务的训练样本
        is_next = False
    return sentence, next_sentence, is_next
def _get_nsp_data_from_paragraph(paragraph, paragraphs, vocab, max_len): # 从输入paragraph中生成用于下一句预测的训练样本
    nsp_data_from_paragraph = []
    for i in range(len(paragraph) - 1): #paragraph里面是一个个句子列表
        tokens_a, tokens_b, is_next = _get_next_sentence(
            paragraph[i], paragraph[i + 1], paragraphs)
        # 考虑1个'<cls>'词元和2个'<sep>'词元3所以要加上三
        if len(tokens_a) + len(tokens_b) + 3 > max_len:
            continue # 序列太长的样本就不要了
        tokens, segments = d2l.get_tokens_and_segments(tokens_a, tokens_b)
        nsp_data_from_paragraph.append((tokens, segments, is_next)) # 生成这些训练样本的bert编码信息
    return nsp_data_from_paragraph

# 生成遮蔽语言模型任务的数据
def _replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_preds, vocab): # tokens表示BERT输入序列的词元的列表，candidate_pred_positions是不包括特殊词元的BERT输入序列的词元索引的列表（特殊词元在遮蔽语言模型任务中不被预测），以及num_mlm_preds指示预测的数量
    # 为遮蔽语言模型的输入创建新的词元副本，其中输入可能包含替换的“<mask>”或随机词元或不变
    mlm_input_tokens = [token for token in tokens]
    pred_positions_and_labels = []
    # 打乱后用于在遮蔽语言模型任务中获取随机词元进行预测
    random.shuffle(candidate_pred_positions)
    for mlm_pred_position in candidate_pred_positions: # 遍历每一个需要掩蔽的位置
        if len(pred_positions_and_labels) >= num_mlm_preds: # 首先判断列表中是否已经存在足够数量的预测词元
            break
        masked_token = None

        if random.random() < 0.8: # 80%的时间：将词替换为“<mask>”词元
            masked_token = '<mask>'
        else:
            if random.random() < 0.5: # 10%的时间：保持词不变
                masked_token = tokens[mlm_pred_position]

            else: # 10%的时间：用随机词替换该词
                masked_token = random.choice(vocab.idx_to_token)
        mlm_input_tokens[mlm_pred_position] = masked_token
        pred_positions_and_labels.append(
            (mlm_pred_position, tokens[mlm_pred_position]))
    return mlm_input_tokens, pred_positions_and_labels # 返回可能替换后的输入词元、发生预测的词元索引和这些预测的标签
def _get_mlm_data_from_tokens(tokens, vocab):
    candidate_pred_positions = []
    # tokens是一个字符串列表
    for i, token in enumerate(tokens):# 遍历每一个词元
        if token in ['<cls>', '<sep>']:# 在遮蔽语言模型任务中不会预测特殊词元，所以除了他们以外的词元被考虑成为候选的掩蔽词元
            continue
        candidate_pred_positions.append(i)
    # 遮蔽语言模型任务中预测15%的随机词元
    num_mlm_preds = max(1, round(len(tokens) * 0.15)) # 得到需要掩蔽的词元个数，用这个max1是为了避免没有需要预测的词元（即上述计算结果为0）导致程序出错
    mlm_input_tokens, pred_positions_and_labels = _replace_mlm_tokens(
        tokens, candidate_pred_positions, num_mlm_preds, vocab)
    pred_positions_and_labels = sorted(pred_positions_and_labels,
                                       key=lambda x: x[0]) # 排序后，pred_positions_and_labels列表中的元素按照位置从小到大排序
    pred_positions = [v[0] for v in pred_positions_and_labels] # 只包含预测词元位置的列表
    mlm_pred_labels = [v[1] for v in pred_positions_and_labels] # 只包含预测词元标签的列表
    return vocab[mlm_input_tokens], pred_positions, vocab[mlm_pred_labels] # 返回输入词元的索引，发生预测的词元索引以及这些预测的标签索引


def _pad_bert_inputs(examples, max_len, vocab): # 将特殊的“<mask>”词元附加到输入。examples包含来自两个预训练任务的辅助函数_get_nsp_data_from_paragraph和_get_mlm_data_from_tokens的输出
    max_num_mlm_preds = round(max_len * 0.15) # 最大掩蔽词元的个数
    all_token_ids, all_segments, valid_lens,  = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []
    for (token_ids, pred_positions, mlm_pred_label_ids, segments,is_next) in examples: # 遍历examples中的元素，将它们的数据填充到最大序列长度max_len，同时将NSP和MLM任务相关的信息提取出来存储到不同的列表中
        all_token_ids.append(torch.tensor(token_ids + [vocab['<pad>']] * (
            max_len - len(token_ids)), dtype=torch.long)) # 将token_ids（输入序列）补全到最大长度的max_len，使用PAD符号填充
        all_segments.append(torch.tensor(segments + [0] * (
            max_len - len(segments)), dtype=torch.long)) # 将segments（输入的segment ids）同样进行补全，使用0填充
        # valid_lens不包括'<pad>'的计数
        valid_lens.append(torch.tensor(len(token_ids), dtype=torch.float32)) # 将segments（输入的segment ids）同样进行补全，使用0填充
        all_pred_positions.append(torch.tensor(pred_positions + [0] * (
            max_num_mlm_preds - len(pred_positions)), dtype=torch.long)) # 对pred_positions（预测词元的位置）进行补全，使得每个例子中预测的词元数量等于整个序列长度的15%（即max_num_mlm_preds）
        # 填充词元的预测将通过乘以0权重在损失中过滤掉
        all_mlm_weights.append(
            torch.tensor([1.0] * len(mlm_pred_label_ids) + [0.0] * (
                max_num_mlm_preds - len(pred_positions)),
                dtype=torch.float32)) # 预测词元的权重向量，如果该位置上是真实的词元，权重为1；否则为0
        all_mlm_labels.append(torch.tensor(mlm_pred_label_ids + [0] * (
            max_num_mlm_preds - len(mlm_pred_label_ids)), dtype=torch.long)) # 补全mlm_pred_label_ids（预测词元的标签），使其数量等于max_num_mlm_preds
        nsp_labels.append(torch.tensor(is_next, dtype=torch.long)) # 将is_next（下一句是否为相邻句子）添加到nsp_labels中
    return (all_token_ids, all_segments, valid_lens, all_pred_positions,
            all_mlm_weights, all_mlm_labels, nsp_labels) # 返回处理后的元组们，补全后的输入序列、segment ids、有效长度、预测词元位置、预测词元权重、预测词元标签以及NSP任务的标签

class _WikiTextDataset(torch.utils.data.Dataset):
    def __init__(self, paragraphs, max_len):
        # 输入paragraphs[i]是代表段落的句子字符串列表；
        # 而输出paragraphs[i]是代表段落的句子列表，其中每个句子都是词元列表
        paragraphs = [d2l.tokenize(
            paragraph, token='word') for paragraph in paragraphs]
        sentences = [sentence for paragraph in paragraphs
                     for sentence in paragraph]
        self.vocab = d2l.Vocab(sentences, min_freq=5, reserved_tokens=[
            '<pad>', '<mask>', '<cls>', '<sep>'])
        # 获取下一句子预测任务的数据
        examples = []
        for paragraph in paragraphs:
            examples.extend(_get_nsp_data_from_paragraph(
                paragraph, paragraphs, self.vocab, max_len))
        # 获取遮蔽语言模型任务的数据
        examples = [(_get_mlm_data_from_tokens(tokens, self.vocab)
                      + (segments, is_next))
                     for tokens, segments, is_next in examples]
        # 填充输入
        (self.all_token_ids, self.all_segments, self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights,
         self.all_mlm_labels, self.nsp_labels) = _pad_bert_inputs(
            examples, max_len, self.vocab)

    def __getitem__(self, idx): #可以任意访问WikiText-2语料库的一对句子生成的预训练样本（遮蔽语言模型和下一句预测）样本
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx], self.all_pred_positions[idx],
                self.all_mlm_weights[idx], self.all_mlm_labels[idx],
                self.nsp_labels[idx])

    def __len__(self):
        return len(self.all_token_ids)

def load_data_wiki(batch_size, max_len):
    """加载WikiText-2数据集"""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('wikitext-2', 'wikitext-2')
    paragraphs = _read_wiki(data_dir)
    train_set = _WikiTextDataset(paragraphs, max_len)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size,
                                        shuffle=True, num_workers=num_workers)
    return train_iter, train_set.vocab

batch_size, max_len = 512, 64
train_iter, vocab = load_data_wiki(batch_size, max_len)

for (tokens_X, segments_X, valid_lens_x, pred_positions_X, mlm_weights_X,
     mlm_Y, nsp_y) in train_iter:
    print(tokens_X.shape, segments_X.shape, valid_lens_x.shape,
          pred_positions_X.shape, mlm_weights_X.shape, mlm_Y.shape,
          nsp_y.shape)
    break















