#coding=utf-8
import os
import pandas as pd
import torch
import torchvision
from d2l import torch as d2l
import torchvision.transforms as transforms
from PIL import Image
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#@save
#d2l.DATA_HUB['banana-detection'] = (
#    d2l.DATA_URL + 'banana-detection.zip',
 #   '5de26c8fce5ccdea9f91267273464dc968d20d72')

# 通过read_data_bananas函数，我们读取香蕉检测数据集。 该数据集包括一个的CSV文件，内含目标类别标签和位于左上角和右下角的真实边界框坐标
def read_data_bananas(is_train=True):
    """读取香蕉检测数据集中的图像和标签"""
    totensor = transforms.ToTensor()
    data_dir = d2l.download_extract('banana-detection')
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train
                             else 'bananas_val', 'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name') # 指定的列被设置为索引
    images, targets = [], []
    for img_name, target in csv_data.iterrows(): # 迭代行，返回每行的索引和行本身的对象
        images.append(totensor(Image.open(
            os.path.join(data_dir, 'bananas_train' if is_train else
            'bananas_val', 'images', f'{img_name}'))))

        # 这里的target包含（类别，左上角x，左上角y，右下角x，右下角y），
        # 其中所有图像都具有相同的香蕉类（索引为0）
        targets.append(list(target))  # 得到所有对应图片和对应标签
    return images, torch.tensor(targets).unsqueeze(1) / 256  #在第一维增加一个维度

#通过使用read_data_bananas函数读取图像和标签，以下BananasDataset类别将允许我们创建一个自定义Dataset实例来加载香蕉检测数据集
class BananasDataset(torch.utils.data.Dataset):
    """一个用于加载香蕉检测数据集的自定义数据集"""
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if
              is_train else f' validation examples'))

    def __getitem__(self, idx):
        return (self.features[idx].float(), self.labels[idx])

    def __len__(self):
        return len(self.features)

# 最后，我们定义load_data_bananas函数，来为训练集和测试集返回两个数据加载器实例。对于测试集，无须按随机顺序读取它
def load_data_bananas(batch_size):
    """加载香蕉检测数据集"""
    train_iter = torch.utils.data.DataLoader(BananasDataset(is_train=True),
                                             batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(BananasDataset(is_train=False),
                                           batch_size)
    return train_iter, val_iter

#读取一个小批量，并打印其中的图像和标签的形状。 图像的小批量的形状为（批量大小、通道数、高度、宽度）它与我们之前图像分类任务中的相同。 标签的小批量的形状为（批量大小，m，5），其中m是数据集的任何图像中边界框可能出现的最大数量
#小批量计算虽然高效，但要求每张图像含有相同数量的边界框，以便放在同一个批量中。 通常来说，图像可能拥有不同数量个边界框；因此，在达到m之前，边界框少于m的图像将被非法边界框填充。 这样，每个边界框的标签将被长度为5的数组表示。 数组中的第一个元素是边界框中对象的类别，其中-1表示用于填充的非法边界框。 数组的其余四个元素是边界框左上角和右下角的（x，y）坐标值（值域在0～1之间）。 对于香蕉数据集而言，由于每张图像上只有一个边界框，因此m=1
batch_size, edge_size = 32, 256
train_iter, _ = load_data_bananas(batch_size)
batch = next(iter(train_iter))  # 从迭代器里取数据
print(batch[0].shape)  # 一个batch中的图片，应该32张,[32, 3, 256, 256],通道3，宽高
print(batch[1].shape)  # 一个batch中的标签，32个[32, 1, 5]一个边界框，5个标签内容

# 展示10幅带有真实边界框的图像
imgs = (batch[0][0:10].permute(0, 2, 3, 1))  # 重排顺序是因为下面画图函数的要求
axes = d2l.show_images(imgs, 2, 5, scale=2) # 2行5列的方式展示
for ax, label in zip(axes, batch[1][0:10]):
    d2l.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w']) # 图上按标签中的信息把框画出来，第一列到4列是四个角的坐标
d2l.plt.show()























