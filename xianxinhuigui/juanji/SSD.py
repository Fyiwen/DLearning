import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
from PIL import Image
import torchvision.transforms as transforms
from yigemubiaojianceshujuji import load_data_bananas

# 类别预测层,能够得到所有锚框的类别信息
def cls_predictor(num_inputs, num_anchors, num_classes): # num_anchors每个像素对应要生成的锚框数，num_classes类别数
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1), # 加1是因为还有背景类
                     kernel_size=3, padding=1) # 这个参数可以保持输入高和宽不变。 这样一来，输出和输入在特征图宽和高上的空间坐标一一对应。 也就是说一张输出特征图，每个像素位置都对应了一坨以输入特征图此像素位置点为中心生成的锚框的预测类别结果。
# 边界框预测层，能够得到所有锚框的4个偏移量信息
def bbox_predictor(num_inputs, num_anchors):
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1) # 4是因为每个边框信息由四个来表示

''''''#试验：构建两个比例不同的特征图，并且假设每个像素分别生成了5个和3个锚框。假设目标类别的数量为10，看经过类别预测层后的形状改变
#def forward(x, block):
#    return block(x)
#Y1 = forward(torch.zeros((2, 8, 20, 20)), cls_predictor(8, 5, 10)) # 假定一个特征图shape（批量2，通道8，高20，宽20）送入类别预测层（通道8，每个像素5锚框，预测10类别）
#Y2 = forward(torch.zeros((2, 16, 10, 10)), cls_predictor(16, 3, 10))
#print(Y1.shape)# (批量大小，通道数，高度，宽度),通道数=5*11或3*11，其他不变
#print(Y2.shape)
''''''


# 为将上两个预测输出连结起来以提高计算效率，把他们转换为更一致的格式
def flatten_pred(pred):
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1) # 先将通道维移到最后一维，在展平成（批量大小，高*宽*通道数）。start_dim=1表示从第一维到最后一维之间的所有维度值乘起来，其他的维度保持不变

def concat_preds(preds):
    return torch.cat([flatten_pred(p) for p in preds], dim=1) # 将Y1Y2转成一致格式后，在维度1上连结
# 经过形状处理尽管Y1和Y2在通道数、高度和宽度方面具有不同的大小，仍然可以在同一个小批量的两个不同尺度上连接这两个预测输出
#print(concat_preds([Y1, Y2]).shape) # （批量大小，Y1高*宽*通道数+Y2高*宽*通道数）



# 以下模块用于将输入特征图的高度和宽度减半
def down_sample_blk(in_channels, out_channels):
    blk = []
    for _ in range(2):
        blk.append(nn.Conv2d(in_channels, out_channels,
                             kernel_size=3, padding=1)) # 不改形状
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(2)) # 高宽减半
    return nn.Sequential(*blk)
# 对于此高和宽减半块的输入和输出特征图，因为1+(3-1)+(3-1)*1+(2-1)*1*1=6，所以输出中的每个单元在输入上都有一个6*6的感受野。因此，高和宽减半块会扩大每个单元在其输出特征图中的感受野
#print(forward(torch.zeros((2, 3, 20, 20)), down_sample_blk(3, 10)).shape) # (2,10,10,10)通道和高宽变了



#以下为基本网络块，用于从输入图像提取特征
def base_net():
    blk = []
    num_filters = [3, 16, 32, 64] # 给出这个网络每层的通道
    for i in range(len(num_filters) - 1): # 串联3个高和宽减半块，并逐步将通道数翻倍
        blk.append(down_sample_blk(num_filters[i], num_filters[i+1]))
    return nn.Sequential(*blk) # 因为blk里面是list，必须用*号进行转化，才能送入这个函数
#print(forward(torch.zeros((2, 3, 256, 256)), base_net()).shape) # （2，64，32，32）

# 继续为下面的完整SSD模型做准备
def get_blk(i): # 调用上面定义的不同模块，助力最终模型搭建
    if i == 0:
        blk = base_net()
    elif i == 1:
        blk = down_sample_blk(64, 128)
    elif i == 4:
        blk = nn.AdaptiveMaxPool2d((1,1))
    else:
        blk = down_sample_blk(128, 128)
    return blk

def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor): # 为每个块定义前向传播
    Y = blk(X)
    anchors = d2l.multibox_prior(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds) # 特征图Y；在当前尺度下根据Y生成的锚框；预测Y的这些锚框的类别和偏移量结果

# 自定义size和ratio，确定为每个特征图中每个像素生成多少个，什么样的锚框
sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],
         [0.88, 0.961]] # size逐渐增大，是因为越大的size要用于较后的特征图，即更接近顶层的特征图，需要检测更大的对象
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) - 1 # 每个像素4个锚框

# 以下为完整的单发多框检测模型，由五个模块组成，每个块生成的特征图既用于生成锚框，又用于预测这些锚框的类别和偏移量
class TinySSD(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        idx_to_in_channels = [64, 128, 128, 128, 128] #所有输入通道参数
        for i in range(5):
            # 相当于赋值语句self.blk_i=get_blk(i)，只不过这样写i可以自己变不用列举
            setattr(self, f'blk_{i}', get_blk(i))  # setattr(object, name, value)用于设置属性
            setattr(self, f'cls_{i}', cls_predictor(idx_to_in_channels[i],
                                                    num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(idx_to_in_channels[i],
                                                      num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5 #初始化，*5是因为有5个模块分别都要产生这些信息
        for i in range(5):
            # getattr(self,'blk_%d'%i)即访问self.blk_i
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, f'blk_{i}'), sizes[i], ratios[i], getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}')) # X每经过一个模块即一次循环都会变，anchors[i]是第i个模块中的特征图产生的锚框信息，cls_preds[i], bbox_preds[i]同理类别和偏移信息
        anchors = torch.cat(anchors, dim=1) # 将整个网络产生的所有锚框信息连接在一起（批量大小，锚框总数，4）
        cls_preds = concat_preds(cls_preds) # 连接预测信息，shape（批量大小，所有模块特征图高*宽*通道数的和）通道数=一个像素锚框数*类别总数
        cls_preds = cls_preds.reshape(cls_preds.shape[0], -1, self.num_classes + 1) # （批量大小，高*宽*一个像素锚框数，类别总数）
        bbox_preds = concat_preds(bbox_preds) # 连接（批量大小，预测信息总数）
        return anchors, cls_preds, bbox_preds  # 整个网络的检测信息输出

#net = TinySSD(num_classes=1) # 假设只有一个类别（不含背景），输入里不用给背景，运算时会自动加上

#X = torch.zeros((32, 3, 256, 256)) # 输入
#anchors, cls_preds, bbox_preds = net(X) #在所有五个尺度下，每个图像总共生成（32^2+16^2+8^2+4^2+1）*4=5444个锚框 # 第一个模块输出特征图32，第二个输出16.。。第五个输出1
#print('output anchors:', anchors.shape) # （1, 5444, 4）
#print('output class preds:', cls_preds.shape) #(32, 5444, 2)，2是因为含背景 (batch_size, num_anchors, num_classes)
#print('output bbox preds:', bbox_preds.shape) # [32, 21776]，5444*4


# 训练
batch_size = 32
train_iter, _ = load_data_bananas(batch_size)
device, net = d2l.try_gpu(), TinySSD(num_classes=1)
trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)
# 目标检测有两种类型的损失。 第一种有关锚框类别的损失：第二种有关正类锚框偏移量的损失，分别定义他们使用的损失函数
cls_loss = nn.CrossEntropyLoss(reduction='none') # 类别损失函数用交叉熵
bbox_loss = nn.L1Loss(reduction='none')  # 偏移量损失用L1，预测值和真实值之差的绝对值

def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks): # 把两种损失结合起来
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    # cls_preds的shape=(batch_size, num_anchors, num_classes)，cls_labels的shape=(batch_size, num_anchors)
    # 二者经过reshape之后变成(batch_size*num_anchors, num_classes)和(batch_size*num_anchors)计算完loss后又reshape成(batch_size, num_anchors)
    cls = cls_loss(cls_preds.reshape(-1, num_classes), cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1) # 对第二个维度求平均值，得到一个形状为 (batch_size,) 的一维张量，其中每个元素表示对应样本的总损失
    bbox = bbox_loss(bbox_preds * bbox_masks,# 掩码变量bbox_masks令负类锚框和填充锚框不参与损失的计算。
                     bbox_labels * bbox_masks).mean(dim=1)   #bbox_preds和bbox_labelsshape都是(batch_size, num_anchors*4)
    return cls + bbox # 最后将锚框类别和偏移量的损失相加，以获得模型的最终损失函数

# 准确率评价分类结果。
def cls_eval(cls_preds, cls_labels):
    # 由于类别预测结果放在最后一维，argmax需要指定最后一维。
    return float((cls_preds.argmax(dim=-1).type(cls_labels.dtype) == cls_labels).sum())  #取出每个样本预测的概率最大的类别，数据类型转换，比较得布尔值结果，对比较结果进行求和，得到预测正确的样本数量

def bbox_eval(bbox_preds, bbox_labels, bbox_masks): # 由于偏移量使用了L1范数损失，我们使用平均绝对误差来评价边界框的预测结果。
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())


# 在训练模型时，我们需要在模型的前向传播过程中生成多尺度锚框（anchors），并预测其类别（cls_preds）和偏移量（bbox_preds）。 然后，我们根据标签信息Y为生成的锚框标记类别（cls_labels）和偏移量（bbox_labels）。 最后，我们根据类别和偏移量的预测和标注值计算损失函数。为了代码简洁，这里没有评价测试数据集。
num_epochs, timer = 20, d2l.Timer()
animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['class error', 'bbox mae'])
net = net.to(device)
for epoch in range(num_epochs):
    # metric[0]训练精确度（由类别预测计算而来）的和，metric[1]训练精确度的和中的示例数
    # metric[2]绝对误差的和（由预测偏移量计算而来），metric[3]绝对误差的和中的示例数
    metric = d2l.Accumulator(4)
    net.train()
    for features, target in train_iter:
        timer.start()
        trainer.zero_grad()
        X, Y = features.to(device), target.to(device)
        # 输入经过模型生成多尺度的锚框，还得到为每个锚框预测类别和偏移量
        anchors, cls_preds, bbox_preds = net(X)
        # 为每个锚框标注类别和偏移量
        bbox_labels, bbox_masks, cls_labels = d2l.multibox_target(anchors, Y) #根据真实边界框，标注这些锚框的偏移量、掩码、类别
        # 根据类别和偏移量的预测和标注值计算损失函数
        l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                      bbox_masks) # cls_preds是预测类别，cls_labels是真实类别标签，bbox_preds, bbox_labels同理偏移量
        l.mean().backward()
        trainer.step()
        metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(),
                   bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                   bbox_labels.numel()) # 正好对应 metric[0] ，1，2，3
    cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]  # 算出平均类别预测精确度和平均偏移绝对误差
    animator.add(epoch + 1, (cls_err, bbox_mae))
print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
print(f'{len(train_iter.dataset) / timer.stop():.1f} examples/sec on '
      f'{str(device)}')
d2l.plt.show()

# 以下在利用训练好的网络做检测，希望能把图像里面所有我们感兴趣的目标检测出来。
# X = Image.open('D:/pycode/xianxinhuigui/data/img/banana.jpg').unsqueeze(0).float() # 读取并调整测试图像的大小，然后将其转成卷积层需要的四维格式形状为 (1, C, H, W)
transform = transforms.ToTensor()
image = Image.open('D:\\pycode\\xianxinhuigui\\data\\img\\banana.jpg')
X = transform(image).unsqueeze(0).float()
img = X.squeeze(0).permute(1, 2, 0).long()  #得到形状为 (H, W, C) 的 3D 张量。这个操作是为了将张量表示的图像转换为常见的 RGB 图像格式。最后使用 long() 函数将张量的数据类型转换为 int64，以便后续显示图像。

def predict(X):
    net.eval()
    anchors, cls_preds, bbox_preds = net(X.to(device))
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1) #对分类器的预测结果进行 softmax 操作，将预测值转换为概率值。这个操作会沿着第二维（即类别维）计算 softmax，得到一个形状为 (batch_size, num_anchors, num_classes) 的张量，其中每个元素表示对应锚框预测为某个类别的概率值。
    output = d2l.multibox_detection(cls_probs, bbox_preds, anchors) #使用multibox_detection函数，我们可以根据锚框及其预测偏移量得到实际的预测边界框。做预测嘛，预测的锚框位置+预测这个锚框偏了多少=真的预测的框的位置。此真是指真的预测，而不是标签的那个真。然后，通过非极大值抑制来移除相似的预测边界框。
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1] # output[0]的shape=(锚框数，6)，i指第i个锚框，6存储此锚框对应的6个信息:类别，置信度，框坐标4。类别不是背景的算有效预测
    return output[0, idx] # 输出有效锚框们的6个信息

output = predict(X) # shape=（有效锚框个数，6）

#筛选所有置信度不低于阈值的边界框，做为最终输出
def display(img, output, threshold):
    d2l.set_figsize((5, 5))
    fig = d2l.plt.imshow(img)
    for row in output:  # 遍历输出的每一个有效预测边界框
        score = float(row[1]) #当前锚框置信度
        if score < threshold:
            continue
        h, w = img.shape[0:2]
        bbox = [row[2:6] * torch.tensor((w, h, w, h), device=row.device)] # 提取预测的框信息后还原到原图
        d2l.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
output = output.to("cpu")
display(img, output, threshold=0.9)
d2l.plt.show()
