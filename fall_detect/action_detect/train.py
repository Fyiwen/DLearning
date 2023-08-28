import torch
from action_detect.data import *
from action_detect.net import *
from torch.utils.data import DataLoader
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import time
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
#DEVICE = "cpu"
# 训练初始全连接网络
class Train:
    def __init__(self, root):

        self.summmaryWriter = SummaryWriter("./logs")  # 将数据以特定的格式存储到这个路径的文件夹中。等实例化writer之后我们使用这个writer对象“拿出来”的任何数据都保存在这个路径之下，之后tensorboard可以可视化这些数据

        # 加载训练数据
        self.train_dataset = PoseDataSet(root, True)  # 此root路径下为训练数据集
        self.train_dataLoader = DataLoader(self.train_dataset, batch_size=100, shuffle=True) # 以批大小为100且随机打乱的方式从训练数据集中读取出来

        # 加载测试数据
        self.test_dataset = PoseDataSet(root, False)  # 测试数据集
        self.test_dataLoader = DataLoader(self.test_dataset, batch_size=512, shuffle=True)

        # 创建模型
        # self.net = NetV1()
        self.net = NetV2()  # 选择定义的netv2网络
        # 加载模型文件（里面的参数），从下面的路径下。相当于把之前预训练得到的参数结果作为这次训练的初始值。再训练，这样结果会针对此特定场景，得到使得模型更好的参数。
        #self.net.load_state_dict(
        #    torch.load("F:/bishe/fall_detect/action_detect/checkPoint/action.pt", map_location='cpu'))
        self.net.to(DEVICE)  # 使用GPU进行训练

        # 定义优化器
        self.opt = optim.SGD(self.net.parameters(),lr=0.0001)  # 这里选择加强版梯度下降法Adam,SGD是普通梯度下降法

    # 启动训练
    def __call__(self):
        for epoch in range(100):
            train_sum_loss = 0  # 总训练损失
            sum_score = torch.tensor(0, dtype=torch.float32)
            for i, (imgs, tags) in enumerate(self.train_dataLoader):  # （x，Y）分别是图像img和标签tag，迭代拿出训练集中每一个内容
                # 将训练集内容添加到GPU
                imgs, tags = imgs.to(DEVICE), tags.to(DEVICE)
                self.net.train()  # 表明现在在训练环境下进行

                train_y = self.net(imgs) # 经过网络得到预测的y值
                loss = torch.mean((tags - train_y) ** 2)  # 计算损失

                self.opt.zero_grad()  # 清空梯度
                loss.backward()  # 反向传播
                self.opt.step()  # 调用优化算法更新模型参数

                train_sum_loss += loss.cpu().detach().item()  # 训练总损失
                predict_targs = torch.argmax(train_y, dim=1)  # 预测的标签是取预测值y的行最大数的下标序号，因为y是有两个数的向量
                label_tags = torch.argmax(tags, dim=1)
                sum_score += torch.eq(predict_targs, label_tags).float().sum().cpu().detach()

            train_avg_loss = train_sum_loss / len(self.train_dataLoader)  # 计算训练平均损失
            train_score = sum_score.item() / len(self.train_dataset)
            # print(epoch,avg_loss)
            #sum_score = 0
            sum_score = torch.tensor(0, dtype=torch.float32)
            test_sum_loss = 0
            #test_accuracy = 0
            for i, (imgs, tags) in enumerate(self.test_dataLoader):
                # 测试集添加到GPU
                imgs, tags = imgs.to(DEVICE), tags.to(DEVICE)

                self.net.eval()  # 标明在测试环境下

                test_y = self.net(imgs)  # 预测值
                loss = torch.mean((tags - test_y) ** 2)  # 损失函数
                test_sum_loss += loss.cpu().detach().item()  # 测试总损失

                predict_targs = torch.argmax(test_y, dim=1)  # 预测的标签是取预测值y的行最大数的下标序号，因为y是有两个数的向量
                label_tags = torch.argmax(tags, dim=1)  #真实的标签是取标签值y的行最大值下标，因为tag是被处理过的one-hot编码有两个数的向量，哪个值大，他的下标对应的就是标签值，要么0要么1
                #sum_score += torch.eq(predict_targs,label_tags).float().cpu().detach().item() #正确的是1错误的是0
                sum_score += torch.eq(predict_targs, label_tags).float().sum().cpu().detach()  # 修改这里
                # accuracy = (outputs.argmax(1) == targets).float().sum().item()
                # test_accuracy += accuracy
            test_avg_loss = test_sum_loss / len(self.test_dataLoader)  # 求平均训练损失
            test_score = sum_score.item()/len(self.test_dataset)
            #test_avg_acc = test_accuracy / len(test_dataset)

            self.summmaryWriter.add_scalars("loss", {"train_avg_loss": train_avg_loss, "test_avg_loss": test_avg_loss},
                                            epoch)  # 可视化时这个变量的名字为loss，要存档其内容，到时候在tensorboar可以查看根据其数据形成的图像
            # self.summmaryWriter.add_scalar("score",score,epoch)
            self.summmaryWriter.add_scalars("accuracy", {"train_accuracy": train_score,"test_accuracy": test_score}, epoch)
            print(epoch, train_avg_loss, test_avg_loss)  # 输出每一轮的平均训练和测试损失
            print(train_score,test_score)

            # 添加时间戳
            now_time = int(time.time())  # 返回当前时间的时间戳

            timeArray = time.localtime(now_time)  # 格式化时间戳为本地的时间

            str_time = time.strftime("%Y-%m-%d-%H:%M:%S", timeArray)  # 把刚才的一大串事件信息格式化成我们想要的呈现形式

            torch.save(self.net.state_dict(), f"F:/bishe/fall_detect/action_detect/checkPoint/train/action.jit")  # 在这个路径下保存训练得到的模型的每一层参数


if __name__ == '__main__':  # 在本页面运行可以进行训练
    train = Train('F:/bishe/fall_detect/data')
    train()
