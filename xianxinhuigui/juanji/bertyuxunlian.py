import torch
from torch import nn
from d2l import torch as d2l
# 有了bert模型和数据集就可以对bert进行预训练

# 首先加载WikiText-2数据集作为小批量的预训练样本，用于遮蔽语言模型和下一句预测
batch_size, max_len = 512, 64
train_iter, vocab = d2l.load_data_wiki(batch_size, max_len)

# 定义了一个小的BERT模型，使用了2层、128个隐藏单元和2个自注意头
net = d2l.BERTModel(len(vocab), num_hiddens=128, norm_shape=[128],
                    ffn_num_input=128, ffn_num_hiddens=256, num_heads=2,
                    num_layers=2, dropout=0.2, key_size=128, query_size=128,
                    value_size=128, hid_in_features=128, mlm_in_features=128,
                    nsp_in_features=128)
devices = d2l.try_all_gpus()
loss = nn.CrossEntropyLoss()

# 下面这个是辅助函数。用于给定训练样本，计算遮蔽语言模型和下一句子预测任务的损失。BERT预训练的最终损失是遮蔽语言模型损失和下一句预测损失的和
def _get_batch_loss_bert(net, loss, vocab_size, tokens_X,segments_X, valid_lens_x,
                         pred_positions_X, mlm_weights_X,mlm_Y, nsp_y):

    _, mlm_Y_hat, nsp_Y_hat = net(tokens_X, segments_X,
                                  valid_lens_x.reshape(-1),
                                  pred_positions_X) # 得到掩蔽位置预测结果和下一句子的二分类预测结果
    # 计算遮蔽语言模型损失
    mlm_l = loss(mlm_Y_hat.reshape(-1, vocab_size), mlm_Y.reshape(-1)) *\
            mlm_weights_X.reshape(-1, 1) # 由于在训练数据中的输入序列长度不一定相等，因此需要通过mlm_weights_X张量来指示哪些位置是真正有意义的，哪些位置是填充的。其形状和mlm_Y_hat相同，来表示是否对应有效的位置
    mlm_l = mlm_l.sum() / (mlm_weights_X.sum() + 1e-8)
    # 计算下一句子预测任务的损失
    nsp_l = loss(nsp_Y_hat, nsp_y)
    l = mlm_l + nsp_l # 得到总损失
    return mlm_l, nsp_l, l

# train_bert函数定义了在WikiText-2（train_iter）数据集上预训练BERT（net）的过程
def train_bert(train_iter, net, loss, vocab_size, devices, num_steps):
    net = nn.DataParallel(net, device_ids=devices).to(devices[0]) # 将神经网络模型在不同的GPU上进行并行处理，从而加速训练过程。同时，通过.to(devices[0])将模型移动到主设备，保证在训练时每个批次的数据只被发送到一个GPU上进行处理。这样能够避免多个GPU之间的通信带来的额外开销，以及可能出现的内存不足的问题
    trainer = torch.optim.Adam(net.parameters(), lr=0.01)
    step, timer = 0, d2l.Timer()
    animator = d2l.Animator(xlabel='step', ylabel='loss',
                            xlim=[1, num_steps], legend=['mlm', 'nsp'])
    # 遮蔽语言模型损失的和，下一句预测任务损失的和，句子对的数量，计数
    metric = d2l.Accumulator(4)
    num_steps_reached = False
    while step < num_steps and not num_steps_reached: # num_steps指定了训练的迭代步数
        for tokens_X, segments_X, valid_lens_x, pred_positions_X,\
            mlm_weights_X, mlm_Y, nsp_y in train_iter:
            tokens_X = tokens_X.to(devices[0])
            segments_X = segments_X.to(devices[0])
            valid_lens_x = valid_lens_x.to(devices[0])
            pred_positions_X = pred_positions_X.to(devices[0])
            mlm_weights_X = mlm_weights_X.to(devices[0])
            mlm_Y, nsp_y = mlm_Y.to(devices[0]), nsp_y.to(devices[0])
            trainer.zero_grad()
            timer.start()
            mlm_l, nsp_l, l = _get_batch_loss_bert(
                net, loss, vocab_size, tokens_X, segments_X, valid_lens_x,
                pred_positions_X, mlm_weights_X, mlm_Y, nsp_y)
            l.backward()
            trainer.step()
            metric.add(mlm_l, nsp_l, tokens_X.shape[0], 1)
            timer.stop()
            animator.add(step + 1,
                         (metric[0] / metric[3], metric[1] / metric[3]))
            step += 1
            if step == num_steps:
                num_steps_reached = True # 达到了指定步数
                break

    print(f'MLM loss {metric[0] / metric[3]:.3f}, '
          f'NSP loss {metric[1] / metric[3]:.3f}')
    print(f'{metric[2] / timer.sum():.1f} sentence pairs/sec on '
          f'{str(devices)}')


# 预训练过程中，可以绘制出遮蔽语言模型损失和下一句预测损失
train_bert(train_iter, net, loss, len(vocab), devices, 50)

# 预训练BERT之后，可以用它来表示单个文本、文本对或其中的任何词元
def get_bert_encoding(net, tokens_a, tokens_b=None): # 函数返回tokens_a和tokens_b中所有词元的BERT表示
    tokens, segments = d2l.get_tokens_and_segments(tokens_a, tokens_b)
    token_ids = torch.tensor(vocab[tokens], device=devices[0]).unsqueeze(0)
    segments = torch.tensor(segments, device=devices[0]).unsqueeze(0)
    valid_len = torch.tensor(len(tokens), device=devices[0]).unsqueeze(0)
    encoded_X, _, _ = net(token_ids, segments, valid_len)
    return encoded_X

# 考虑“a crane is flying”这句话
tokens_a = ['a', 'crane', 'is', 'flying']
encoded_text = get_bert_encoding(net, tokens_a)
# 词元：'<cls>','a','crane','is','flying','<sep>'
encoded_text_cls = encoded_text[:, 0, :] # 以cls位置的bert表示代表整个输入语句的BERT表示
encoded_text_crane = encoded_text[:, 2, :] # 得到crane位置的bert表示
print(encoded_text.shape, encoded_text_cls.shape, encoded_text_crane[0][:3])

# 考虑一个句子“a crane driver came”和“he just left”
tokens_a, tokens_b = ['a', 'crane', 'driver', 'came'], ['he', 'just', 'left']
encoded_pair = get_bert_encoding(net, tokens_a, tokens_b)
# 词元：'<cls>','a','crane','driver','came','<sep>','he','just','left','<sep>'
encoded_pair_cls = encoded_pair[:, 0, :]
encoded_pair_crane = encoded_pair[:, 2, :]
print(encoded_pair.shape, encoded_pair_cls.shape, encoded_pair_crane[0][:3])











