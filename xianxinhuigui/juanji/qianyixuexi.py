import os
import torch
import torchvision
from torch import nn
from d2l import torch as d2l
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# 将在一个小型数据集上微调ResNet模型。该模型已在ImageNet数据集上进行了预训练
#将数据下载到本地，然后解压
#d2l.DATA_HUB['hotdog'] = (d2l.DATA_URL + 'hotdog.zip','fba480ffa8aa7e0febbb511d181409f899b9baa5')
# 解压数据集
data_dir = d2l.download_extract('hotdog')

train_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'))
test_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'))

# 显示前8个正类样本图片和最后8张负类样本图片
hotdogs = [train_imgs[i][0] for i in range(8)]
not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]
d2l.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4)  # 两行八列

# 使用RGB通道的均值和标准差，以标准化每个通道
normalize = torchvision.transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#训练期间，我们首先从图像中裁切随机大小和随机长宽比的区域，然后将该区域缩放为输入图像
train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224), # 将给定图像随机裁剪为不同的大小和宽高比，然后缩放所裁剪得到的图像为制定的大小
    torchvision.transforms.RandomHorizontalFlip(), # 以给定的概率随机水平旋转给定的PIL的图像，默认为0.5
    torchvision.transforms.ToTensor(),
    normalize])
# 测试过程中，我们将图像的高度和宽度都缩放到256像素，然后裁剪中央区域作为输入
test_augs = torchvision.transforms.Compose([
    torchvision.transforms.Resize([256, 256]),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    normalize])
#使用在ImageNet数据集上预训练的ResNet-18作为源模型。 在这里，我们指定pretrained=True以自动下载预训练的模型参数
pretrained_net = torchvision.models.resnet18(pretrained=True)
#pretrained_net.fc就是源网络的连接层

# 目标模型finetune_net中成员变量features的参数被初始化为源模型相应层的模型参数,成员变量output的参数是随机初始化的
finetune_net = torchvision.models.resnet18(pretrained=True)
finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)  # 前面的网络直接从源移植，最后一层主动修改
nn.init.xavier_uniform_(finetune_net.fc.weight)  # 随机初始化这个全连接层

#定义了一个训练函数train_fine_tuning，该函数使用微调，因此可以多次调用。如果param_group=True，输出层中的模型参数将使用十倍的学习率
def train_fine_tuning(net, learning_rate, batch_size=8, num_epochs=5,
                      param_group=True):
    train_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train'), transform=train_augs),
        batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'test'), transform=test_augs),
        batch_size=batch_size)
    devices = d2l.try_all_gpus()
    loss = nn.CrossEntropyLoss(reduction="none")
    if param_group:
        params_1x = [param for name, param in net.named_parameters()
             if name not in ["fc.weight", "fc.bias"]]  # 除了最后一个全连接层的参数以外的其他参数，在训练时按照小学习率更新
        trainer = torch.optim.SGD([{'params': params_1x},
                                   {'params': net.fc.parameters(),# 全连接层的参数用大学习率更新
                                    'lr': learning_rate * 10}],
                                lr=learning_rate, weight_decay=0.001)
    else:
        trainer = torch.optim.SGD(net.parameters(), lr=learning_rate,
                                  weight_decay=0.001)
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
                   devices)

#使用较小的学习率，通过微调预训练获得的模型参数
train_fine_tuning(finetune_net, 5e-5)

#进行比较，定义了一个相同的模型，但是将其所有模型参数初始化为随机值。 由于整个模型需要从头开始训练，因此需要使用更大的学习率
scratch_net = torchvision.models.resnet18()
scratch_net.fc = nn.Linear(scratch_net.fc.in_features, 2)
train_fine_tuning(scratch_net, 5e-4, param_group=False)