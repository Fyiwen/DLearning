import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
import torchvision.transforms.functional

# 使用在ImageNet数据集上预训练的ResNet-18模型来提取图像特征
pretrained_net = torchvision.models.resnet18(pretrained=True)
print(list(pretrained_net.children())[-3:]) #打印出网络每层信息

net = nn.Sequential(*list(pretrained_net.children())[:-2]) # 把最后的两层去掉,新网络只采用前面的那些层。这里调用的是children，有一个类似的modules调用，区别注意一下

X = torch.rand(size=(1, 3, 320, 480)) # 定义一个输入
print(net(X).shape)
# 这边形状为[1, 512, 10, 15]

num_classes = 21
net.add_module('final_conv', nn.Conv2d(512, num_classes, kernel_size=1)) # 增加1*1卷积层。512是因为上面print看到最后一层的输出通道为512，这里直接输出通道选成类别数，只是为了方便算，当然可以换成别的
net.add_module('transpose_conv', nn.ConvTranspose2d(num_classes, num_classes,
                                    kernel_size=64, padding=16, stride=32)) # 增加反卷积层，使得输入输出形状一样
#关于这个参数怎么来的，可以这么算，因为经过前面网络一坨卷积后，现在输出为10*15，那么利用卷积计算输出的公式10=(320-k+2p+s)/s,15=(420-k+2p+s)/s,再加上一个规律，可以推测k=64，p=16，s=32
# 现在的net就是完整的全连接卷积神经网络

'''下面介绍用双线性插值法初始化的转置卷积层'''
def bilinear_kernel(in_channels, out_channels, kernel_size): # 这个函数的实现，用于为转置卷积生成双线性插值核
    factor = (kernel_size + 1) // 2 # 内核的中心位置
    if kernel_size % 2 == 1:  # 确保内核是对称的
        center = factor - 1
    else:
        center = factor - 0.5
    og = (torch.arange(kernel_size).reshape(-1, 1),  # 第一个张量表示网格的 y 坐标，而第二个张量表示同一网格的 x 坐标
          torch.arange(kernel_size).reshape(1, -1))
    filt = (1 - torch.abs(og[0] - center) / factor) * \
           (1 - torch.abs(og[1] - center) / factor)  # 此公式根据每个像素到内核中心的距离为其分配权重，从而为靠近中心的像素提供更高的权重。得到双线性滤波器
    weight = torch.zeros((in_channels, out_channels,
                          kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt # 通过将双线性滤波器值分配给张量中的适当位置来更新张量。过滤器值被分配给张量的对角线元素
    return weight
conv_trans = nn.ConvTranspose2d(3, 3, kernel_size=4, padding=1, stride=2,bias=False) # 将输入的高和宽放大2倍的转置卷积层
conv_trans.weight.data.copy_(bilinear_kernel(3, 3, 4)) # 将这个转置卷积层的卷积核用bilinear_kernel函数初始化。copy_可以改变原来的张量，clone（）就只会产生一个复制值不改变原来

img = torchvision.transforms.ToTensor()(d2l.Image.open('D:/pycode/xianxinhuigui/data/img/catdog.jpg')) # 把图片转成张量
X = img.unsqueeze(0) # 增加维度，现在变成适合送进网络的4维（b，c，h，w）
Y = conv_trans(X) # 经过转置卷积层。相当于做了有关双线性插值的上采样
out_img = Y[0].permute(1, 2, 0).detach() # 为了显示上采样后的图片，先改形状
#打印采样前后的图片看一下
d2l.set_figsize()
print('input image shape:', img.permute(1, 2, 0).shape) # 原图形状
d2l.plt.imshow(img.permute(1, 2, 0))
print('output image shape:', out_img.shape) # 上采样后形状
d2l.plt.imshow(out_img)
''''''

#利用上面那个插值法初始化一个适合现在这个全卷积神经网络的转置卷积层的核
W = bilinear_kernel(num_classes, num_classes, 64)
net.transpose_conv.weight.data.copy_(W) # 把这个新核换上去


# 训练
batch_size, crop_size = 32, (320, 480)
train_iter, test_iter = d2l.load_data_voc(batch_size, crop_size)

def loss(inputs, targets):
    return F.cross_entropy(inputs, targets, reduction='none').mean(1).mean(1)#在像素级预测任务中，通常会生成一个损失张量，其中每个像素位置都有相应的损失值。通过对高度维度进行平均操作，我们可以获得每个样本在高度方向上的平均损失值。考虑到在像素级预测任务中，每个样本都是图像，具有高度和宽度这两个维度。因此，对高度维度进行平均操作有助于将每个样本的损失值降低为一个标量值，即该样本在高度方向上的平均损失。通过对高度维度进行平均操作后，我们还可以选择继续对宽度维度进行平均操作，即mean(1)的第二次使用。这将进一步降低维度，得到每个样本在高度和宽度方向上的平均损失值。

num_epochs, lr, wd, devices = 5, 0.001, 1e-3, d2l.try_all_gpus()
trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)

#预测
def predict(img):
    X = test_iter.dataset.normalize_image(img).unsqueeze(0)# 将输入图像在各个通道做标准化，并转成卷积神经网络所需要的四维输入格式
    pred = net(X.to(devices[0])).argmax(dim=1) #1就是类别维度
    return pred.reshape(pred.shape[1], pred.shape[2]) # pred.shape[1]图像的高，2为宽

def label2image(pred): # 将预测类别转换成颜色
    colormap = torch.tensor(d2l.VOC_COLORMAP, device=devices[0]) # 将颜色映射表转换为张量
    X = pred.long() # 将输入的预测标签pred转换为整数类型（取整）并赋值给X。这是因为colormap张量中的索引必须是整数
    return colormap[X, :]# 根据类别都找到了对应颜色

voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')
test_images, test_labels = d2l.read_voc_images(voc_dir, False)
n, imgs = 4, []
# 由于每个图片大小各异。模型使用了步幅为32的转置卷积层，因此当输入图像的高或宽无法被32整除时，转置卷积层输出的高或宽会与输入图像的尺寸有偏差。 为了解决这个问题，我们可以在图像中截取多块高和宽为32的整数倍的矩形区域，并分别对这些区域中的像素做前向传播。 请注意，这些区域的并集需要完整覆盖输入图像。 当一个像素被多个区域所覆盖时，它在不同区域前向传播中转置卷积层输出的平均值可以作为softmax运算的输入，从而预测类别。这里方便起见，不用这么完美的方法，直接截取一部分符合的大小预测
for i in range(n):
    crop_rect = (0, 0, 320, 480) # 这里采用从左上角开始截取形状为320*480的区域用于预测
    X = torchvision.transforms.functional.crop(test_images[i], *crop_rect) # 裁剪每张图片
    pred = label2image(predict(X)) # 得到预测后的颜色图像
    imgs += [X.permute(1,2,0), pred.cpu(), # 打印截取的图像，对应预测的颜色图像
             torchvision.transforms.functional.crop( # 原图对应颜色标签图片一样做裁剪后，显示出来
                 test_labels[i], *crop_rect).permute(1,2,0)]
d2l.show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n, scale=2) # imgs[::3]跳着取imgs里面图片，去到0，3，6.。正好是所有原图。imgs[1::3]从1开始跳着取正好是所有预测图