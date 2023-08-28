# 处理Lei2数据集为了LSTM
import torch
import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader

# 训练LSTM网络
from action_detect.net import *
from torch.utils.data import DataLoader
from torchvision import transforms

def load_video(video_path,frame_size=(224, 224)):   # 把视频分帧
    cap = cv2.VideoCapture(video_path)  # 截出所有帧
    frames = []
    while True:
        ret, frame = cap.read()  # 读取每一帧
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        #frame=np.array(frame)
        # Resize video
        frame = cv2.resize(frame, frame_size)  # 调整帧大小
        # Normalize video
        frame = frame / 255.0

        frames.append(frame)  # 所有帧放入一个对象
    cap.release()
    return np.array(frames)


class Lei2Dataset(Dataset):
    def __init__(self, root1_dir,root2_dir ,transform=None):
        self.root1_dir = root1_dir  # 视频路径
        self.root2_dir = root2_dir  #txt路径
        self.transform = transform
        self.video_files = os.listdir(root1_dir)  # 根目录下所有视频文件
        self.txt_files = os.listdir(root2_dir)  # 根目录下所有视频文件
    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):  # # 从对应的文本文件中读取起始帧和结束帧（如果存在）。否则，我们将起始帧和结束帧设置为0。然后根据它们创建标签。最后返回视频和标签。
        # Load video
        video_path = os.path.join(self.root1_dir, self.video_files[idx]) # 合并路径
        video = load_video(video_path)  # 所有帧都在这里

        txt_path=os.path.join(self.root2_dir, self.txt_files[idx])
        if os.path.isfile(txt_path):
            with open(txt_path, "r") as f:
                lines = f.readlines()
                start_frame = int(lines[0])  # 读txt文件前两行
                end_frame = int(lines[1])
            label=1
        else:
            start_frame = 0
            end_frame = len(video)
            label=0
        # 将视频中跌倒状态的帧起点和终点与帧图像组成序列
        sequence = video[start_frame:end_frame]
        # 可以在这里进行数据增强等操作
        if self.transform:
            sequence = [self.transform(frame) for frame in sequence]
        # 将帧序列和标签打包成一个元组，方便在DataLoader中使用
        label_onehot = torch.zeros(2)
        label_onehot[label] = 1
        sample = (sequence,label_onehot.long())
        return sample  # 返回跌倒序列和跌倒标签




DEVICE = "cpu"
transform = transforms.Compose([
    transforms.ToTensor(),
])
# 定义一个batch的连续帧数
batch_size = 6
frame_count = 4

train_dataset = Lei2Dataset(root1_dir='F:/bishe/fall_detect/data/Lei2/Coffee_room_01/Videos',root2_dir='F:/bishe/fall_detect/data/Lei2/Coffee_room_01/Annotation_files', transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#for batch in train_dataloader:  # 来将每个batch中的帧序列打包成序列张量的。具体来说，它将一个batch中的连续的frame_count帧打包成一个可变长度的序列张量，然后将打包后的张量存储在一个列表中。打包后的张量将成为LSTM模型的输入。
   # batch_frames = [pack_sequence(batch[i:i+frame_count])
                  #  for i in range(batch_size - frame_count)]
test_dataset =Lei2Dataset(root1_dir='F:/bishe/fall_detect/data/Lei2/Home_02/Videos',root2_dir='F:/bishe/fall_detect/data/Lei2/Home_02/Annotation_files',transform=None)
test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=True, num_workers=4)
input_size = 1
hidden_size = 128
num_layers = 2
output_size = 2

model = LSTM(input_size, hidden_size, num_layers, output_size)
model.to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_dataloader):
        data, targets = data.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, batch_idx+1, len(train_dataloader), loss.item()))




