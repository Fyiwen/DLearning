import os
import cv2
import numpy as np
from torch.utils.data import Dataset

class PoseDataSet(Dataset):  #继承了pytorch的Dataset，当获得一个数据集后此类帮助我们提取想要的信息

    def __init__(self, root, is_train = True):  # 此函数用于对象实例化，通常用来提供类中需要使用的变量。这里root会传入数据集所在的根目录，默认情况下是训练模式
        super(PoseDataSet, self).__init__() # 调用父类方法Dataset对所有变量初始化
        self.dataset = []  # 初始化读取操作

        sub_dir = "train" if is_train else "test" # 如果is_train是true即现在是训练模式，那么这条路径名就是train否则test

        for tag in os.listdir(f"{root}/{sub_dir}"):  # listdir方法会将路径下的所有文件名（包括后缀名）组成一个列表。这里root=F:/bishe/ism_person_openpose-master/data是数据集的位置，
                                                      # 其下的train或test(以下都已train来解释）目录下的文件名只有两个fall和normal，分别赋给tag
            file_dir = f"{root}/{sub_dir}/{tag}" # 此路径是F:/bishe/ism_person_openpose-master/data/train/fall或者F:/bishe/ism_person_openpose-master/data/train/normal
            for img_file in os.listdir(file_dir): # 此路径下的所有文件就是姿态图片，摔倒的和正常的，他们的文件名一一赋给img_path
                img_path = f"{file_dir}/{img_file}" # 此路径直接定位到某张姿态图片
                if tag == 'fall':  # 如果是摔倒类型的姿态图片
                    self.dataset.append((img_path, 0))  # 把跌倒图片标记为0
                else:  # 如果是正常姿态的图片
                    self.dataset.append((img_path, 1))  # 把正常图片标记为1
                # print(self.dataset)
# 经过init，fall和normal文件下的图片都已经读入数据集，
    def __len__(self):  #重写的方法，返回数据集的大小

        return len(self.dataset)

    def __getitem__(self, item):  # 重写的方法，此方法用于实例化时根据索引值获取每一个图片数据并且获取其对应的Label
        data = self.dataset[item]  # 根据传入的item值（下标索引，框架自己在用dataloader读数据时自动提供），去读图片，获得第item张图片的存储地址和label
        img = cv2.imread(data[0],cv2.IMREAD_GRAYSCALE) # 以灰度图形式读取图像，data[0]是此图片的存储地址
        img = img.reshape(-1) # 将图像信息直接拉成一行多列的数据，因为我们忽略了空间结构， 所以我们使用reshape将每个二维图像转换为一个长度为num_inputs的向量
        img = img/255 # 把图像数据进行归一化，范围是[0,1]，转成标准模式，数值范围缩小，便于训练
    # tag——one-hot操作目的是吧标记的0，1转为one-hot编码
        tag_one_hot = np.zeros(2)  # 初始化数组([0,0])
        tag_one_hot[int(data[1])] = 1  # data[1]存放的是这个图片的标签，强制类型转换后的数值作为下表，此下标对应位置置1

        return np.float32(img),np.float32(tag_one_hot)  # 将img和tag_one_hot的数据转成float32的类型后返回,比如实例化train_loader = DataLoader(PoseDataSet, ....)之后for X, y in train_loader就可以读出return返回的两个值了,分别对应x，y


if __name__ == '__main__':  # 以下的代码，只有在本文件作为脚本直接执行时才会被执行，而import到其他脚本中是不会被执行的
    dataset = PoseDataSet('F:/bishe/fall_detect/data')  # 将此路径下的内容作为数据集进行读取，root=F:/bishe/ism_person_openpose-master/data
    print(dataset[0][1])  #理解为dataset.__getitem__(0)[1]，0是第1张图片的item值，1是data【1】标签，至于第一张图片是谁有可能按照固定顺序，有可能shuffle=true打乱了


