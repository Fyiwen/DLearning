import os
import torch
import torchvision
from d2l import torch as d2l
import torchvision.transforms.functional as F
# 这是一个比较重要的语义分割数据集Pascal VOC2012，下载下来
d2l.DATA_HUB['voc2012'] = (d2l.DATA_URL + 'VOCtrainval_11-May-2012.tar',
                           '4e443f8a2eca6b1dac8a6c57641b67dd40621a49')

voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012') # 提取出的数据集位于../data/VOCdevkit/VOC2012

def read_voc_images(voc_dir, is_train=True): # 函数定义为利用txt文件，将所有对应的输入的图像和标签读入内存
    """读取所有VOC图像并标注"""
    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation',
                             'train.txt' if is_train else 'val.txt')  # ImageSets/Segmentation路径包含用于训练和测试样本的文本文件，而JPEGImages和SegmentationClass路径分别存储着每个示例的输入图像和标签
    mode = torchvision.io.image.ImageReadMode.RGB #它指示 torchvision.io.read_image() 函数将图像读取为RGB模式
    with open(txt_fname, 'r') as f: # 打开了名为 txt_fname 的文本文件，打开的文件对象被赋值给变量 f，语句创建了一个上下文环境，在此内可以对文件进行读取和操作，不需要手动关闭文件
        images = f.read().split() # 一个字符串列表，其中每个元素都是文件中的一行
    features, labels = [], []
    for i, fname in enumerate(images): # 得到每一张样本图片的名字
        features.append(torchvision.io.read_image(os.path.join(
            voc_dir, 'JPEGImages', f'{fname}.jpg'))) # 读取对应路径下，当前文件名的图片
        labels.append(torchvision.io.read_image(os.path.join( # 读取对应路径下，当前文件名的标签图片
            voc_dir, 'SegmentationClass' ,f'{fname}.png'), mode)) # 因为语义分割需要一个像素一个label，所以label还是存成一张图片，注意png比较好，不压缩。所以训练时是按照一张图片，他的标号也是一张对应图片，标签中颜色相同的像素属于同一个语义类别
    return features, labels # 返回提取到的图片和对应标签图

train_features, train_labels = read_voc_images(voc_dir, True) # 读取到训练图片和标签

n = 5
imgs = train_features[0:n] + train_labels[0:n] # 读五个图片和他对应的标签图片
imgs = [img.permute(1,2,0) for img in imgs] # 对每个图进行处理，因为要显示图像，显示时形状最好是通道在最后，所以要改变一下位置。一般读取时通道在前
d2l.show_images(imgs, 2, n)

# 这个说明了RGB的颜色，3个数字为一组，表示一个颜色
VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

# 这个是RGB颜色对应的类
VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']
# 这边相当于构建
def voc_colormap2label():
    """构建从RGB到VOC类别索引的映射"""#【，，】->number
    colormap2label = torch.zeros(256 ** 3, dtype=torch.long) # 初始化256^3
    for i, colormap in enumerate(VOC_COLORMAP): # 遍历每一组颜色
        colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i  # 自定义了一种从一组3个数字表示的颜色，映射到类别索引数字的方式.将颜色编码为一个整数
    return colormap2label # 返回所有映射信息

# 这边相当于查询
def voc_label_indices(colormap, colormap2label): # 真的给了一个颜色后，利用上面函数定义好的，找到他的类别
    """将VOC标签中的RGB值映射到它们的类别索引"""
    colormap = colormap.permute(1, 2, 0).numpy().asype('int32') #将标签图片的格式改成通道在最后，因为后续需要按照像素的顺序遍历图像，并且每个像素都是以 (x, y)的形式表示的，其中x和y分别对应于行和列，因此需要将通道维度调整到最后。使用 numpy() 方法将张量转换为 NumPy 数组，然后使用 astype 方法将数据类型转换为 int32，以便后续的计算
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2]) # 对每一个像素，将其 RGB 颜色值编码成一个整数，用于找类别。正好图片的三个通道对应3个RGB颜色，分别对每个颜色通道上的像素进行对应处理
    return colormap2label[idx] # 得到对应类别，这里colormap2label就是上面voc_colormap2label


y = voc_label_indices(train_labels[0], voc_colormap2label()) #将第一张样本标签图片，本来用的颜色表示类别，现在将每一个像素转换成直观的类别索引0，1
print(y[105:115, 130:140], VOC_CLASSES[1]) #在第一张样本图像中，可以看到飞机头部区域的类别索引为1，而背景索引为0

def voc_rand_crop(feature, label, height, width):
    """随机裁剪特征和标签图像""" # 注意裁剪图片后也要对相应的标签图片进行裁剪
    rect = torchvision.transforms.RandomCrop.get_params(
        feature, (height, width)) # 获取裁剪参数，对图片按照这个宽高比进行随即裁剪。返回一个四元组 (top, left, height, width)，表示裁剪框的位置和大小
    feature = torchvision.transforms.functional.crop(feature, *rect) # *rect 表示裁剪框的位置和大小，这里使用了可变参数语法，将四元组 (top, left, height, width) 展开成四个独立的参数
    label = torchvision.transforms.functional.crop(label, *rect)
    return feature, label

imgs = []
for _ in range(n):
    imgs += voc_rand_crop(train_features[0], train_labels[0], 200, 300) # 按照200*300的比例随即裁剪第一张样本和标签，裁剪出n个存在imgs里面

imgs = [img.permute(1, 2, 0) for img in imgs] # 把每张图的通道移到最后一维
d2l.show_images(imgs[::2] + imgs[1::2], 2, n) # 显示裁剪后的图片，将列表 imgs 中的所有图像两两配对，并将它们沿着第一维度拼接起来，形成一个新的列表。这个操作相当于将所有偶数索引位置上的图像和所有奇数索引位置上的图像配对，然后按照顺序依次拼接起来。2 表示每行显示两个图像，n 表示总共显示的图像数量也可以看作n行
# imgs[::2] 表示从列表 imgs 的第一个元素开始，每隔一个元素取一个元素，直到列表末尾，得到一个新的子列表，其中包含了 imgs 中所有索引位置为偶数的元素
# imgs[1::2] 表示从列表 imgs 的第二个元素开始，每隔一个元素取一个元素，直到列表末尾，得到一个新的子列表，其中包含了 imgs 中所有索引位置为奇数的元素
# + 表示将上述两个子列表拼接起来，得到一个新的列表。拼接的方法是先将 imgs[::2] 中的所有元素依次排列在前面，然后将 imgs[1::2] 中的所有元素依次排列在后面，形成一个新的列表。这个操作相当于将所有偶数索引位置上的元素和所有奇数索引位置上的元素配对，然后按照顺序依次拼接起来
# 本来一个图一个标签再一图一标签，现在变成n个图跟n个标签

class VOCSegDataset(torch.utils.data.Dataset):
    """一个用于加载VOC数据集的自定义数据集"""

    def __init__(self, is_train, crop_size, voc_dir):
        self.transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])# 对图像数据进行标准化处理。它的作用是对每个颜色通道进行均值方差归一化，使得图像的像素值在每个通道上都服从均值为0、标准差为1的正态分布。这种标准化操作可以提高模型的稳定性和收敛速度，从而提高模型的准确率。这些参数值是在 ImageNet 数据集上统计得到的。因此，这个 Normalize 操作是用来将输入图像标准化为 ImageNet 数据集上的标准分布，从而方便使用 ImageNet 上预训练的模型进行迁移学习。
        self.crop_size = crop_size # 裁剪大小
        features, labels = read_voc_images(voc_dir, is_train=is_train) # 读取训练图片和标签
        self.features = [self.normalize_image(feature) # 给所有筛选后可用的训练图片做标准化
                         for feature in self.filter(features)]
        self.labels = self.filter(labels)
        self.colormap2label = voc_colormap2label() # 存定义好的RGB-》类别的映射关系
        print('read ' + str(len(self.features)) + ' examples')

    def normalize_image(self, img): #对输入图像的RGB三个通道的值分别做标准化
        return self.transform(img.float() / 255)

    def filter(self, imgs): # 这个函数用于如果输入的图片的高宽小于要裁剪的大小，就直接不要了，不采用只返回那些大的图
        return [img for img in imgs if (
            img.shape[1] >= self.crop_size[0] and
            img.shape[2] >= self.crop_size[1])]

    def __getitem__(self, idx): #可以任意访问数据集中索引为idx的输入图像及其每个像素的类别索引
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx], # idx是在这个列表中随机选取的一个下标
                                       *self.crop_size) # 给一对图片-标签做随即裁剪然后作为训练内容输出
        return (feature, voc_label_indices(label, self.colormap2label)) # 返回的是图和类别标签，RGB的不好计算，虽然存的时候存RGB但是训练时类别索引好算

    def __len__(self):
        return len(self.features)


'''这一坨是分散型得到训练集的数据迭代'''
crop_size = (320, 480) # 裁剪大小
voc_train = VOCSegDataset(True, crop_size, voc_dir) # 读取出来的训练集
voc_test = VOCSegDataset(False, crop_size, voc_dir) # 读取出来的测试集

batch_size = 64
train_iter = torch.utils.data.DataLoader(voc_train, batch_size, shuffle=True,drop_last=True,
                                    num_workers=d2l.get_dataloader_workers()) # drop_last=True 表示如果最后一个 batch 的样本数量小于 batch size，那么这些样本将被忽略。num_workers=d2l.get_dataloader_workers() 表示使用多个 worker 来读取数据。在 PyTorch 中，数据加载器默认只使用一个 worker 来读取数据，也就是说，在每个 batch 中，数据加载器每次只能读取一个样本。使用多个 worker 可以并行地读取数据，从而提高数据读取的效率。d2l.get_dataloader_workers() 是一个自定义函数，用于根据当前系统的 CPU 核心数来选择合适的 worker 数量
# 打印出用于训练的样本的最终形状
for X, Y in train_iter:
    print(X.shape)
    print(Y.shape)
    break


'''这一坨是搞了个函数得到训练集的数据迭代'''
# 以上内容是为了了解这个数据集。实际上定义了上面那些函数后使用以下函数就可以读取下载的Pascal VOC2012语义分割数据集。返回训练集和测试集的数据迭代
def load_data_voc(batch_size, crop_size):
    """加载VOC语义分割数据集"""
    voc_dir = d2l.download_extract('voc2012', os.path.join(
        'VOCdevkit', 'VOC2012')) # 数据集路径
    num_workers = d2l.get_dataloader_workers()
    train_iter = torch.utils.data.DataLoader(
        VOCSegDataset(True, crop_size, voc_dir), batch_size,
        shuffle=True, drop_last=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(
        VOCSegDataset(False, crop_size, voc_dir), batch_size,
        drop_last=True, num_workers=num_workers)
    return train_iter, test_iter

