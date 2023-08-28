import torch
import torchvision
from torch import nn
from d2l import torch as d2l

d2l.set_figsize()
content_img = d2l.Image.open('D:/pycode/xianxinhuigui/data/img/rainier.jpg') # 读取内容图像
d2l.plt.imshow(content_img)

style_img = d2l.Image.open('D:/pycode/xianxinhuigui/data/img/autumn-oak.jpg') # 读取风格图像
d2l.plt.imshow(style_img)

# 归一化要用的RGB参数
rgb_mean = torch.tensor([0.485, 0.456, 0.406])
rgb_std = torch.tensor([0.229, 0.224, 0.225])

def preprocess(img, image_shape):  # 预处理函数，将图片变成张量,并且转换成适合输入的形状
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_shape),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=rgb_mean, std=rgb_std)])
    return transforms(img).unsqueeze(0)

def postprocess(img):  #后处理函数postprocess则将输出图像中的像素值还原回标准化之前的值。 由于图像打印函数要求每个像素的浮点数值在0～1之间，我们对小于0和大于1的值分别取0和1 。主要做从张量变成图片
    img = img[0].to(rgb_std.device)
    img = torch.clamp(img.permute(1, 2, 0) * rgb_std + rgb_mean, 0, 1) # 将像素值还原回标准化之前。torch.clamp函数用于将结果限制在范围[0, 1]之间，确保图像的像素值在有效范围内
    return torchvision.transforms.ToPILImage()(img.permute(2, 0, 1)) # 将处理后的图像张量转换为PIL图像对象，方便后面保存或显示图像

pretrained_net = torchvision.models.vgg19(pretrained=True) # 使用基于ImageNet数据集预训练的VGG-19模型来抽取图像特征
# 样式层选这几层是因为又想得到局部信息，又想得到全局信息来匹配。层数越往下是细节，往上更全局
style_layers, content_layers = [0, 5, 10, 19, 28], [25] # 选择VGG中部分卷积层用于提取样式和内容特征
net = nn.Sequential(*[pretrained_net.features[i] for i in
                      range(max(content_layers + style_layers) + 1)]) # 使用出VGG中上面选的那几层中最大层索引加一范围内的层组建新网络，其他的层不要


#因为在训练时无须改变预训练的VGG的模型参数，所以我们可以在训练开始之前就提取出内容特征和风格特征。 使用下面的get_contents，get_styles
# 由于合成图像是风格迁移所需迭代的模型参数，我们只能在训练过程中通过调用extract_features函数来抽取合成图像的内容特征和风格特征
def extract_features(X, content_layers, style_layers):
    contents = [] # 存储每个内容层的输出
    styles = [] # 存储每个样式层的输出
    for i in range(len(net)): # 逐层计算
        X = net[i](X)
        if i in style_layers: # 保留内容层和风格层的输出
            styles.append(X)
        if i in content_layers:
            contents.append(X)
    return contents, styles

def get_contents(image_shape, device): # 对内容图像抽取内容特征。
    content_X = preprocess(content_img, image_shape).to(device) # 预处理内容图像
    contents_Y, _ = extract_features(content_X, content_layers, style_layers) # 送入网络后得到所有内容层的输出
    return content_X, contents_Y

def get_styles(image_shape, device):# 对风格图像抽取风格特征
    style_X = preprocess(style_img, image_shape).to(device) # 预处理风格图像
    _, styles_Y = extract_features(style_X, content_layers, style_layers) # 送入网络后得到所有风格层的输出
    return style_X, styles_Y


# 开始计算损失函数
def content_loss(Y_hat, Y): # 平方误差函数的两个输入均为extract_features函数计算所得到的内容层的输出，假设Y已经预先算出
    return torch.square(Y_hat - Y.detach()).mean()

def gram(X): # X是风格层的输出，形状为（1，c，h，w）
    num_channels, n = X.shape[1], X.numel() // X.shape[1] # num_channels=c，n=h*w
    X = X.reshape((num_channels, n)) # x现在是矩阵c*hw
    return torch.matmul(X, X.T) / (num_channels * n) # 求出拉格姆矩阵xXT，矩阵中的每一个元素xij是xi和xj的内积表达了通道i和通道j上风格特征的相关性。由于当n的值较大时，格拉姆矩阵中的元素容易出现较大的值，为了让风格损失不受这些值的大小影响，下面定义的gram函数将格拉姆矩阵除以了矩阵中元素的个数，即

def style_loss(Y_hat, gram_Y): # 假设gram_Y已经预先算出，一样输入是extract_features函数计算所得到的风格层的输出，但是风格层的输出样本数为1，通道数为c高和宽分别为h和w，将此输出转换成矩阵x，用上面的gram函数
    return torch.square(gram(Y_hat) - gram_Y.detach()).mean() # gram后使用格拉姆矩阵来表达风格层输出的风格，再求平方误差

# 全变分损失，因为学到的合成图像里面有大量高频噪点，即有特别亮或者特别暗的颗粒像素。 一种常见的去噪方法是全变分去噪，降低全变分损失能够尽可能使邻近的像素值相似。
def tv_loss(Y_hat):# 就是计算Y_hat中相邻元素之间的差异的平均值
    return 0.5 * (torch.abs(Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).mean() + # 计算Y_hat在第三个维度即高度维度相邻元素之间的差异。它通过从第一个元素开始到倒数第二个元素结束，分别与其下一个元素进行相减操作。所有批次、所有通道的元素，但在高度维度上，它选择从索引为1到最后一个索引的所有元素
                  torch.abs(Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).mean()) #在第四个维度上（即宽度维度）相邻元素之间的差异。平均值乘以0.5。这是为了对两个方向（高度和宽度）的差异值进行平均，以得到整体的差异程度。乘以0.5是为了保持数值范围一致，以便与其他损失或指标进行合理的比较

content_weight, style_weight, tv_weight = 1, 1e3, 10  # 比例权重，代表他们各自的损失在合并损失时的重要程度？？？？？？？？

def compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram):
    # 分别计算内容损失、风格损失和全变分损失
    contents_l = [content_loss(Y_hat, Y) * content_weight for Y_hat, Y in zip( # zip函数将contents_Y_hat和contents_Y中对应位置的元素打包成一个个元组
        contents_Y_hat, contents_Y)]  # 所有内容损失，将内容损失值乘以权重，以便在计算总损失时考虑其贡献度。
    styles_l = [style_loss(Y_hat, Y) * style_weight for Y_hat, Y in zip(
        styles_Y_hat, styles_Y_gram)] # 所有风格损失
    tv_l = tv_loss(X) * tv_weight # 所有全变分损失
    # 对所有损失求和
    l = sum(10 * styles_l + contents_l + [tv_l]) # 加权和。 通过调节那些权重超参数，我们可以权衡合成图像在保留内容、迁移风格以及去噪三方面的相对重要性。
    return contents_l, styles_l, tv_l, l

# 训练期间，可以通过优化算法来更新合成图像，以使其更好地匹配预定的目标内容和风格
class SynthesizedImage(nn.Module): #合成的图像是训练期间唯一需要更新的变量。因此定义一个简单的模型SynthesizedImage，将合成的图像视为模型参数。模型的前向传播只需返回模型参数即可
    def __init__(self, img_shape, **kwargs): # img_shape是合成图像的形状，**kwargs表示接受任意数量的关键字参数，用于传递额外的参数
        super(SynthesizedImage, self).__init__(**kwargs) # 调用父类的初始化方法
        self.weight = nn.Parameter(torch.rand(*img_shape)) #创建了一个nn.Parameter对象，表示模型的可训练参数（合成图像），即这个weight，这里初始化了一下他按照合成图像的尺寸随机初始化

    def forward(self):  #定义当调用模型实例时要执行的计算过程
        return self.weight # 返回模型参数self.weight，也就是合成图像。由于合成图像是模型的唯一需要更新的变量，因此前向传播只需返回它即可。

def get_inits(X, device, lr, styles_Y):  # 该函数创建了合成图像的模型实例，并将其初始化为内容图片。风格图像在各个风格层的格拉姆矩阵styles_Y_gram将在训练前预先计算好
    gen_img = SynthesizedImage(X.shape).to(device) # 创建了一个合成图像的模型实例gen_img
    gen_img.weight.data.copy_(X.data) # 将图像X的数据复制到合成图像模型的参数gen_img.weight中。这样，合成图像模型的初始值就与输入图像X相同。也就是没训练前合成图像和原内容图片一样
    trainer = torch.optim.Adam(gen_img.parameters(), lr=lr) # 优化器，gen_img.parameters()返回合成图像模型中的可训练参数
    styles_Y_gram = [gram(Y) for Y in styles_Y] #计算了风格图像经过各个风格层的输出的格拉姆矩阵
    return gen_img(), styles_Y_gram, trainer

# 训练
def train(X, contents_Y, styles_Y, device, lr, num_epochs, lr_decay_epoch):
    X, styles_Y_gram, trainer = get_inits(X, device, lr, styles_Y) # 现在x为合成图像模型实例
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_decay_epoch, 0.8) # 创建了一个StepLR学习率调度器。它接受三个参数：优化器对象trainer、学习率衰减的间隔（以训练时期计）lr_decay_epoch和衰减因子0.8。StepLR调度器会在每个指定的间隔（以训练时期计）调整优化器的学习率，将其乘以衰减因子。
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[10, num_epochs],
                            legend=['content', 'style', 'TV'],
                            ncols=2, figsize=(7, 2.5))
    for epoch in range(num_epochs):
        trainer.zero_grad()
        contents_Y_hat, styles_Y_hat = extract_features(
            X, content_layers, style_layers) # 不断抽取合成图像的内容特征和风格特征，计算损失，迭代模型
        contents_l, styles_l, tv_l, l = compute_loss(
            X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram)
        l.backward() # 计算损失函数对于模型参数的梯度
        trainer.step() # 优化器会根据当前的参数值和梯度信息来更新模型的参数
        scheduler.step() # 更新优化器的学习率
        if (epoch + 1) % 10 == 0:
            animator.axes[1].imshow(postprocess(X))
            animator.add(epoch + 1, [float(sum(contents_l)),
                                     float(sum(styles_l)), float(tv_l)])
    return X

device, image_shape = d2l.try_gpu(), (300, 450)
net = net.to(device)
content_X, contents_Y = get_contents(image_shape, device)
_, styles_Y = get_styles(image_shape, device)
output = train(content_X, contents_Y, styles_Y, device, 0.3, 500, 50) # 训练后得到最终的合成图像
