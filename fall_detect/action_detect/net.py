from torch import nn, randn, exp, sum
import torch
from torch import nn
import torch.nn.functional as F
import math


# 定义了两种网络
class NetV1(nn.Module):

    def __init__(self):
        super().__init__()  # 初始化所有变量

        self.W = nn.Parameter(randn(16384, 2))  # 网络中的权值w先随机初始化为某一个数

    # 前项过程逻辑
    def forward(self, x):  # 网络的输入值为x
        h = x @ self.W  # h=x1w1+x2w2。。@代表矩阵乘法
        # 以下是手动具体实现了softmax函数内容
        h = exp(h) # 这里求指数，下面求和，最后相除
        z = sum(h, dim=1, keepdim=True)  # 保持梯度
        return h / z


class NetV2(nn.Module):  # 这是一个分类网络，全连接网络，输入16384维的图片（拉直后的图片）然后输出两个类别分数，即有两个值的向量
    def __init__(self):
        super().__init__()

        self.sequential = nn.Sequential(  # 这个网络层是一个线性层，一个激活函数，又一个线性层，然后sotfmax函数
            nn.Linear(16384, 100),
            nn.ReLU(),
            nn.Linear(100, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.sequential(x)

# LSTM 网络
'''
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
    

        batch_size, seq_len, c, h, w = x.shape
        x = x.reshape(batch_size * seq_len, c, h, w)
        h0 = torch.zeros(self.num_layers, batch_size * seq_len, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size * seq_len, self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        

        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
'''
'''
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(3 * input_size * input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, sequence_length, channels, height, width = x.shape
        x = x.view(batch_size * sequence_length, channels, height, width)
        _, (h_n, _) = self.lstm(x)
        x = self.fc(h_n[-1, :, :])
        x = self.sigmoid(x)
        return x
'''
'''
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(3 * input_size * input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, sequence_length, channels, height, width = x.shape
        x = x.view(batch_size * sequence_length, channels, height, width)
        x = x.unsqueeze(1) # 增加一个维度
        _, (h_n, _) = self.lstm(x)
        x = self.fc(h_n[-1, :, :])
        x = self.sigmoid(x)
        return x

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(3 * input_size , hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size, sequence_length, channels = x.shape
        x = x.view(batch_size, sequence_length, -1)
        x = x.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(x)
        x = self.fc(h_n[-1])
        x = self.softmax(x)  # 修改这里
        return x

'''
'''
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.conv = nn.Sequential(
            nn.Conv2d(input_size, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.lstm = nn.LSTM(256, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.conv(x)
        batch_size, time_steps, channels, height, width = x.shape
        x = x.view(batch_size, time_steps, -1)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
'''
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, time_steps, channels, height, width = x.shape
        x = x.reshape(batch_size, time_steps, -1)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out

'''
class TransformerNet(nn.Module):
    def __init__(self, num_classes):
        super(TransformerNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.transformer_layer = nn.Transformer(nhead=1, num_encoder_layers=1)

    def forward(self, x):
        # 输入形状：(batch_size, frames_per_clip, 3, 图像高度, 图像宽度)
        x = x.permute(1, 0, 2, 3, 4).contiguous()
        # 转换形状为：(frames_per_clip, batch_size, 3, 图像高度, 图像宽度)
        b, f, c, h, w = x.size()
        x = x.view(b * f, c, h, w)
        # 卷积神经网络
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        src = x.view(b, f, -1)  # 定义源数据张量
        tgt = torch.zeros_like(src).to(x.device)  # 定义目标数据张量
        # 使用 nn.Linear 层将 src 和 tgt 的最后一维大小从当前值转换为 512
        src = self.fc2(src)
        tgt = self.fc2(tgt)
        # Transformer层
        x = self.transformer_layer(src, tgt)  # 传递源数据和目标数据张量
        # 转换形状为：(batch_size, embedding_size, frames_per_clip)
        x = x.permute(1, 2, 0).contiguous()
        # 最大池化
        x = F.max_pool1d(x, x.size(2))
        # 转换形状为：(batch_size, embedding_size)
        x = x.squeeze(2)
'''


class TransformerNet(nn.Module):
    def __init__(self, frames_per_clip, img_height, img_width,channels):
        super(TransformerNet, self).__init__()
        self.frames_per_clip = frames_per_clip
        self.img_height = img_height
        self.img_width = img_width

        # 定义卷积层
        self.conv1 = nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)

        # 定义Transformer Encoder层
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=16, nhead=1)

        # 定义全连接层
        self.fc1 = nn.Linear(16 * self.img_height * self.img_width, 32)
        self.fc2 = nn.Linear(32, 2)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        batch_size, frames_per_clip, channels, height, width = x.shape
        # 将输入的张量形状转换为(batch_size*frames_per_clip, channels, height, width)
        x = x.reshape(batch_size * frames_per_clip, channels, height, width)
        # 卷积层
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # 将卷积层输出的张量形状转换为(batch_size*frames_per_clip, 64, height*width)
        x = x.reshape(batch_size * frames_per_clip, 16, height * width)
        # Transformer Encoder层
        x = x.permute(2, 0, 1)
        x = self.transformer_encoder_layer(x)
        x = x.permute(1, 2, 0)
        # 将张量形状转换为(batch_size*frames_per_clip, 64*height*width)
        x = x.reshape(batch_size * frames_per_clip, 16 * self.img_height * self.img_width)
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # 将输出的张量形状转换为(batch_size, frames_per_clip, 2)
        x = x.reshape(batch_size, self.frames_per_clip, 2)
        x = x[:, -1, :]
        x = self.sigmoid(x)
        return x










