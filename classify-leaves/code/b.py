import torch
import torch.nn as nn
from d2l import torch as d2l
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import datetime
from torch.utils.tensorboard import SummaryWriter  # 导入tensorboard
from torchvision import transforms
from torch.nn import functional as F
import matplotlib.pyplot as plt
import torchvision.models as models
# This is for the progress bar.
from tqdm import tqdm
import seaborn as sns
"""
数据预处理
训练集train.csv中：图片序号+标签、测试集test.cvs中：图片序号。27153张图片文件，其中0~18352为训练集，剩余为测试集
"""

label_dataframe = pd.read_csv('../train.csv')  # 将训练集表格读入
print(label_dataframe.head(5))  # 查看前五行的内容

  # 此函数可以生成描述性统计数据，统计数据集的集中趋势，分散和行列的分布情况，不包括 NaN值
print(label_dataframe.describe())  # 输出中可以看到训练集总共有18353张图片，标签有176类

leaves_labels=sorted(list(set(label_dataframe['label'])))  # 获取训练集标签并按照字母顺序进行排序，转变成set集合就是去重过程
n_classes = len(leaves_labels)  # 获取标签长度即标签个数
print(n_classes)
print(leaves_labels[:10])  # 输出十个标签

class_to_num = dict(zip(leaves_labels, range(n_classes)))  # 把标签转换成数字表示。这两个方法可以将两个列表合并成一个一一对应的字典
print(class_to_num)

num_to_class = {v : k for k, v in class_to_num.items()}  # 数字转换成类别名称的字典，便于最后预测用。类别算在k，数字算在v

class LeavesData(Dataset):  # 继承pytorch的dataset，创建树叶数据集类，用于批量管理训练集，验证集和测试集
    def __init__(self, csv_path, file_path, mode='train', valid_ratio=0.2,resize_height=256, resize_width=256):
        """
        :param csv_path: csv文件路径
        :param file_path: 图像文件所在路径
        :param mode: 训练、验证还是测试模式
        :param valid_ratio: 验证集在整个train.csv中的比例
        :param resize_height:调整后的图片高
        :param resize_width:调整后的图片宽
        """
        self.resize_height = resize_height  # 将大小不同的照片，调整成一个尺寸
        self.resize_width = resize_width
        self.file_path = file_path
        self.mode = mode

        self.data_info =pd.read_csv(csv_path, header=None)  # 读取csv文件，并且去掉表头部分
        self.data_len = len(self.data_info.index)-1  # 数据集长度
        self.train_len = int(self.data_len*(1-valid_ratio))  # 训练数据集长度
        if mode == 'train': # 训练数据集和验证数据集由train.csv文件拆开而来
            self.train_image = np.asarray(self.data_info.iloc[1:self.train_len, 0])  # 读取第一列（是图片名称），从第二行读到最后一张训练图片位置。转换为数组
            self.train_label = np.asarray(self.data_info.iloc[1:self.train_len, 1])  # 读取第二列（是图片标签），从第二行读到最后一张训练图片位置
            self.image_arr = self.train_image
            self.label_arr = self.train_label
        elif mode == 'valid':
            self.valid_image = np.asarray(self.data_info.iloc[self.train_len:, 0])
            self.valid_label = np.asarray(self.data_info.iloc[self.train_len:,1])
            self.image_arr = self.valid_image
            self.label_arr = self.valid_label
        elif mode == 'test':  # 测试集来自另外一个test.cvs文件
            self.test_image = np.asarray(self.data_info.iloc[1:,0])  # 读取测试集图像名称列的所有名称
            self.image_arr = self.test_image
        self.real_len = len(self.image_arr)
        print('finished reading the {} set of leaves Dataset({} samples found)'.format(mode, self.real_len))  # 提示输出表示读完哪个数据集的一共多少个样例

    def __getitem__(self, index):  # 接受一个索引返回一个样本
        single_image_name = self.image_arr[index]  # 从image_arr中得到索引对应的图像名
        img_as_img = Image.open(self.file_path + single_image_name)  # 读取图像名对应的图像文件，通过这个open读取的图片是PIL类型
        # 如果需要将RGB三通道的图片转换成灰度图片可参考下面两行
        if img_as_img.mode != 'L':
            img_as_img = img_as_img.convert('L')
        if self.mode == 'train':  # 定义一系列transform，包括将图片大小放缩到给定大小224*224，0.5概率随机水平翻转，将一个PIL 格式的图片转换成tensor格式的图片
            transform = transforms.Compose([transforms.Resize((224, 224)),transforms.RandomHorizontalFlip(p=0.5),transforms.ToTensor()])  # 用于组合一系列的变换操作，比如选择一个概率做随机水平翻转
        else:  # valid和test不做数据增强，所以另外写
            transform =transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor()])
        img_as_img = transform(img_as_img)  # 保存transform后的图像
        if self.mode == 'test':# 测试集中没有标签所以返回的少
            return img_as_img
        else:
            label = self.label_arr[index]  # 根据索引得到图像的字符串标签
            number_label = class_to_num[label]  # 得到字符串标签的对应数字
            return img_as_img, number_label  # 返回每一个index对应的图片数据和对应的标签
    def __len__(self):  # 决定了读取的长度
        return self.real_len

train_path = '../train.csv'
test_path = '../test.csv'
img_path = '../'

train_dataset = LeavesData(train_path, img_path, mode='train')
val_dataset = LeavesData(train_path, img_path,mode='valid')
test_dataset = LeavesData(test_path,img_path,mode='test')
print(train_dataset)
print(val_dataset)
print(test_dataset)

# 以下为数据集们的加载器，自动将数据分割成多个小组，训练时对for循环每次抛出一组，顺序随机打乱这里没有，采用5个的多进程读取机制
train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=5
    )
val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=5
    )
test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=5
    )
"""
# 给大家展示一下数据长啥样
def im_convert(tensor):
    
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1,2,0)
    image = image.clip(0, 1)

    return image

fig=plt.figure(figsize=(20, 12))
columns = 4
rows = 2

dataiter = iter(val_loader)
inputs, classes = dataiter.next()

for idx in range (columns*rows):
    ax = fig.add_subplot(rows, columns, idx+1, xticks=[], yticks=[])
    ax.set_title(num_to_class[int(classes[idx])])
    plt.imshow(im_convert(inputs[idx]))
plt.show()
# 看一下是在cpu还是GPU上
def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

device = get_device()
print(device)
"""
"""
def set_parameter_requires_grad(model, feature_extracting):  # 是否要冻住模型的前面一些层
    if feature_extracting:
        model = model
        for param in model.parameters():
            param.requires_grad = False
# resnet34模型
def res_model(num_classes, feature_extract = False, use_pretrained=True):
    model_ft = models.resnet34(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc=nn.Sequential(nn.Linear(num_ftrs, num_classes))
    return model_ft
# 超参数
learning_rate =3e-4
weight_decay =1e-3
num_epoch=50
model_path='./pre_res_model.ckpt'
# 初始化模型并且放在指定设备
model=res_model(176)
model=model.to(device)
model.device=device
# 交叉熵作为损失函数
criterion=nn.CrossEntropyLoss()
# 初始化优化器
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay)

n_epochs = num_epoch
best_acc=0.0
for epoch in range(n_epochs):
    model.train()  # 确保在训练模式下
    train_loss=[]  # 用于记录训练中的信息
    train_accs=[]
    for batch in tqdm(train_loader):  # 批量迭代训练集
        imgs,labels=batch  # 批次由图像数据和相应的标签组成。
        # 传播数据。（确保数据和模型位于同一设备上。
        imgs=imgs.to(device)
        labels=labels.to(device)
        
        logits=model(imgs)
        loss=criterion(logits, labels)  # 计算损失，在计算交叉熵之前，我们不需要应用 softmax，因为它是自动完成的。
        optimizer.zero_grad()  # 应首先清除上一步中存储在参数中的梯度。
        loss.backward()  # 计算参数的梯度。
        optimizer.step()  # 使用计算的梯度更新参数。
        acc = (logits.argmax(dim=-1)==labels).float().mean()  # 计算当前批次的准确性。
        train_loss.append(loss.item())  # 记录损失和准确率
        train_accs.append(acc)
    train_loss = sum(train_loss)/len(train_loss)  # # 训练集的平均损失和准确率是记录值的平均值。
    train_acc= sum(train_accs)/len(train_accs)
    print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")  # 输出信息

    model.eval()  # 验证模式下
    # These are used to record information in validation.
    valid_loss = []
    valid_accs = []

    # Iterate the validation set by batches.
    for batch in tqdm(val_loader):
        imgs, labels = batch
        # We don't need gradient in validation.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(imgs.to(device))

        # We can still compute the loss (but not the gradient).
        loss = criterion(logits, labels.to(device))

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        valid_loss.append(loss.item())
        valid_accs.append(acc)

    # The average loss and accuracy for entire validation set is the average of the recorded values.
    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)

    # Print the information.
    print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

    # if the model improves, save a checkpoint at this epoch
    if valid_acc > best_acc:
        best_acc = valid_acc
        torch.save(model.state_dict(), model_path)
        print('saving model with acc {:.3f}'.format(best_acc))

saveFileName = '/content/drive/MyDrive/classify-leaves/submission.csv'

model = res_model(176)

# 从检查点创建模型和加载权重
model = model.to(device)
model.load_state_dict(torch.load(model_path))

# Make sure the model is in eval mode.
# 某些模块（如 Dropout 或 BatchNorm）会影响模型是否处于训练模式。
model.eval()

# 初始化列表以存储预测。
predictions = []
# 按批处理迭代测试集。
for batch in tqdm(test_loader):
    imgs = batch
    with torch.no_grad():
        logits = model(imgs.to(device))

    # 以对数最大的类作为预测并记录下来。
    predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())

preds = []
for i in predictions:
    preds.append(num_to_class[i])

test_data = pd.read_csv(test_path)
test_data['label'] = pd.Series(preds)
submission = pd.concat([test_data['image'], test_data['label']], axis=1)
submission.to_csv(saveFileName, index=False)
print("Done!!!!!!!!!!!!!!!!!!!!!!!!!!!")

"""
# Lenet网络
net = nn.Sequential(  # 实例化一个Sequential块并将需要的层连接在一起
    nn.Conv2d(1, 6, kernel_size=5), nn.Sigmoid(),  # 卷积层，因为用了灰度图所以输入通道1，RGB用3，到这里图片尺寸为220*220
    nn.AvgPool2d(kernel_size=2),  # 平均池化层，到这里图片尺寸为110*110
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),  # 经过这里图片尺寸为106*106
    nn.AvgPool2d(kernel_size=2),  # 经过这里图片尺寸53*53
    nn.Flatten(),  # 此时的输出为4维所以要经过这一层展平输入到多层感知机
    nn.Linear(16 * 53 * 53, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 176))  # 全连接层，因为最后分类的类别有176
# resnet网络准备
class Residual(nn.Module):  # 一个残差网络层，包含两个卷积层和一个旁路支路
    def __init__(self, input_channels, num_channels,use_1x1conv=False, strides=1):
        super().__init__()
        #  一个残差块包含两个卷积层，第一个卷积层通常改变输入输出通道数，并且改变输出的尺寸的形状大小，第二个卷积层输入输出通道数通常不会改变，每一个卷积层会跟着一个批量规范层
        self.conv1 = nn.Conv2d(input_channels, num_channels,kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,kernel_size=3, padding=1)
        if use_1x1conv:  # 当输入与输出通道数不同后需要加一个1x1卷积层，来改变输入X的形状大小和通道数，以便后边与Y相加
            self.conv3 = nn.Conv2d(input_channels, num_channels,kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))# 输入x先经过卷积层，再经过归一化层，在经过激活函数
        Y = self.bn2(self.conv2(Y))  # 再经过第二个卷积层然后再经过归一化层
        if self.conv3: # 如果有用1*1卷积则输入需要经过此层，才能与Y相加
            X = self.conv3(X)
        Y += X  # 将输入X经过两层卷积层得到的输出Y与输入X相加后，再经过ReLU()激活函数,（必须保证X和Y的通道数和尺寸形状大小相同，才能相加）
        return F.relu(Y)

def resnet_block(input_channels, num_channels, num_residuals,first_block=False):                                                                # 相当于上面是一个小残差块，这个看书是把类似的小块组成大块，然后下面bi是很多个大块
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:  #使用1*1卷积层
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:  # 不使用1*1卷积层
            blk.append(Residual(num_channels, num_channels))
    return blk

b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))  # 第一个块
b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))  # 第二个块，因为这里输入输出通道数相同，所以通过把这个参数设为True而调用不添加1*1卷积层的方法
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))

# resnet完整网络
net1 = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(), nn.Linear(512, 176))

# 训练
def inti_weights(m):  # 初始化权重参数
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
# 读取数据
batch_size = 16
train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_iter = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
                                                                                                                                                 #logdir = os.path.join("leavesLog", datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
writer = SummaryWriter('../statistic1')  # SummaryWriter的作用就是，将数据以特定的格式存储到这个路径的文件夹中。首先我们实例化writer之后我们使用这个writer对象“拿出来”的任何数据都保存在这个路径之下，之后tensorboard可以可视化这些数据

# 定义训练函数，里面包括把数据移到GPU，初始化权重，定义优化器，损失函数
def train(net, train_iter, valid_iter, num_epoch, lr, device):
    net.to(device)
    net.apply(inti_weights)  # 逐层初始化
    optimizer = torch.optim.Adam(net.parameters(), lr, weight_decay=0.001)  # 优化器
    lossfn = nn.CrossEntropyLoss()  # 损失函数
    lossfn.to(device)
    for epoch in range(num_epoch):
        print('------ the {} train------'.format(epoch + 1))
        net.train()  # 现在是训练模式
        total_train_loss = 0
        train_accuracy = 0
        train_num = 0
        for i, (X, y) in enumerate(train_iter):  # 迭代拿出训练集中每一个内容
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)  # 复制到同一设备
            y_hat = net(X)  # 预测值
            loss = lossfn(y_hat, y)  # 计算损失
            total_train_loss += loss
            # y_hat是一个二维的张量，行是batch_size,列是每中类别的值（没有sigmoid),取值中最大的索引，就是预测类别，判断与真实类别是否一样
            accuracy = (y_hat.argmax(1) == y).float().sum().item()  # 给定一个类别的预测概率分布y_hat，我们把预测概率最大的类别作为输出类别。如果它与真实类别y一致，说明这次预测是正确的，用float把bool结果转为0，1
            train_accuracy += accuracy
            train_num += 1
            loss.backward()
            optimizer.step()
        print('train {} loss {}'.format(epoch + 1, total_train_loss))
        print('train {} accuracy {}'.format(epoch + 1, train_accuracy / len(train_dataset)))
        net.eval()
        total_test_loss = 0
        test_accuracy = 0
        with torch.no_grad():
            for i, (X, y) in enumerate(valid_iter):
                X, y = X.to(device), y.to(device)
                y_hat = net(X)
                loss = lossfn(y_hat, y)
                total_test_loss += loss
                accuracy = (y_hat.argmax(1) == y).float().sum().item()
                test_accuracy += accuracy
        print("test {} loss{}".format(epoch+1, total_test_loss))
        print("test {}, accuracy {}".format(epoch+1, test_accuracy / len(val_dataset)))
        writer.add_scalars('loss', {'train_loss': total_train_loss / len(train_dataset),
                                    'test_loss': total_test_loss / len(val_dataset)}, epoch)# 可视化时这个变量的名字为loss，要存档内容，到时候在tensorboar可以查看图像
        writer.add_scalars('accuracy', {'train_accuracy': train_accuracy / len(train_dataset),
                                        'test_accuracy': test_accuracy / len(val_dataset)}, epoch)
# 训练
lr = 1e-4
num_epoch = 50
device = torch.device('cuda')
# 可以选用不同的网络
train(net1, train_iter, valid_iter, num_epoch=num_epoch, lr=lr, device=device)  # 这边参数用net就是lenet网络，用net1就是resnet网络
writer.close() # 去激活完pytorch环境下，在命令行tensorboard --logdir=D:\pycode\classify-leaves\statistic，然后得到网址http://localhost:6006/去查看图形