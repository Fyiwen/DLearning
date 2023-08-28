# 处理Lei2数据集为了LSTM
import numpy as np
import os
import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchvision.io.video as video
from torchvision import transforms
from PIL import Image
'''
def load_video(video_file,frame_size=(224, 224)): # 具体路  # 把视频分帧
    cap = cv2.VideoCapture(video_file)  # 截出所有帧
    # 帧的总数和宽高
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frames = []
    while True:
        ret, frame = cap.read()  # 读取每一帧
        if not ret:
            break
        frame = cv2.resize(frame, frame_size)  # 调整帧大小
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Normalize video
        #frame = frame.reshape(-1)
        frame = frame / 255.0
       # frame = np.transpose(frame, (2, 0, 1))
        frames.append(frame)  # 所有帧放入一个对象
    cap.release()
    #frames = np.array(frames)
    return frames,num_frames
'''
'''
def load_video(video_file, input_size):
    cap = cv2.VideoCapture(video_file)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (input_size, input_size))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame / 255.0
        frame = frame.reshape(input_size, input_size, -1)  # 现在是input_size x input_size x 3的三维张量
        frames.append(frame)
    cap.release()
    frames = np.array(frames)
    return frames,num_frames
'''
def load_video(video_file, input_size):
    cap = cv2.VideoCapture(video_file)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (input_size, input_size))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame / 255.0
        #frame = Image.fromarray(frame.astype('uint8'), 'RGB')  #将帧转换为PIL Image对象
        #frame = transforms.ToTensor()(frame)  #将PIL Image对象转换为PyTorch张量
        #frame = frame.unsqueeze(0) # tensor用这个frame的形状为(1, 3, input_size, input_size)，三个通道
        frame = torch.from_numpy(frame)  # numpy.ndarray类型的数据转换为tensor
        frame = frame.permute(2, 0, 1)  # (3, input_size, input_size)
        frame = frame.numpy()  # 再转回去
        frame = np.expand_dims(frame, axis=0)  # numpy用这个(1, 3, input_size, input_size)
        frames.append(frame)
    cap.release()
    frames = np.concatenate(frames, axis=0)  # 如果使用NumPy，可以使用np.concatenate函数将所有张量连接起来，(num_frames, 3, input_size, input_size)的四维张量。
    #frames = torch.cat(frames, dim=0)  # 将所有张量连接起来，从而得到形状为(num_frames, 3, input_size, input_size)
    return frames, num_frames

def read_txt(txt_path):#具体路
    """
    读取注释文件并返回跌倒的起始帧数和结束帧数。
    """
    with open(txt_path, 'r') as f:
        start_frame = int(f.readline().strip())
        end_frame = int(f.readline().strip())
    return start_frame, end_frame

def process_data(video_path, txt_path, frames_per_clip, step_between_clips, input_size):
    for filename in os.listdir(video_path):
        if filename.endswith(".avi"):
            video_file = os.path.join(video_path, filename)
            annotation_file = os.path.join(txt_path, filename.replace(".avi", ".txt"))
            frames, num_frames = load_video(video_file, input_size) # (num_frames, 3, input_size, input_size)
            start_frame, end_frame = read_txt(annotation_file)
            clips = []
            if start_frame == 0:
                for i in range(0, num_frames - frames_per_clip, step_between_clips):
                    clip = np.array(frames[i:i+frames_per_clip]) # (frames_per_clip, 3, input_size, input_size)
                   # clip = torch.from_numpy(clip).float()
                    #clip = clip.transpose(3, 0, 1, 2)  # (3, frames_per_clip, input_size, input_size)
                    #clip = clip.permute(1, 0, 2, 3)# (frames_per_clip, 3, input_size, input_size)
                    # clip = clip.reshape(-1, frames_per_clip, 3, input_size, input_size) # (1, frames_per_clip, 3, input_size, input_size)
                    yield clip, 0
            else:
                for i in range(start_frame, end_frame - frames_per_clip, step_between_clips):
                    clip = np.array(frames[i:i+frames_per_clip])
                    #clip = torch.from_numpy(clip).float()
                    #clip = clip.transpose(3, 0, 1, 2)
                    #clip = clip.permute(1, 0, 2, 3)
                    yield clip, 1
                for i in range(0, start_frame - frames_per_clip, step_between_clips):
                    clip = np.array(frames[i:i+frames_per_clip])
                    #clip = torch.from_numpy(clip).float()
                    #clip = clip.transpose(3, 0, 1, 2)
                    #clip = clip.permute(1, 0, 2, 3)
                    yield clip, 0


class FallDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, video_path, txt_path, frames_per_clip, step_between_clips, input_size):
        self.samples = list(process_data(video_path, txt_path, frames_per_clip, step_between_clips, input_size))
        #self.transform = transforms.Compose(
        #    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        clip, label = self.samples[index]
        #clip = self.transform(clip)
        clip = torch.from_numpy(clip).float()
        return clip, label

'''
class Lei2Dataset(Dataset):
    def __init__(self, avi_dir, txt_dir,frames_per_clip, step_between_clips):
        super().__init__()
        self.avi_dir = avi_dir # 差一步视频地址
        self.txt_dir = txt_dir # 差一步txt地址
        # self.files = os.listdir(self.avi_dir)  # 视频文件名字们
        self.frames_per_clip = frames_per_clip
        self.step_between_clips = step_between_clips
        self.clips, self.labels = process_data(avi_dir, txt_dir, frames_per_clip, step_between_clips)

    def __len__(self):
        return len(self.clips) #整个数据集长度

    def __getitem__(self, index):

        clip = self.clips[index] # 提取每一个序列
        label = self.labels[index] # 提取每一个标签
        clip = torch.stack(clip, dim=0) # 这行代码是将采样的一段连续帧序列按照维度0（即第一维）进行堆叠，形成一个新的张量。这个新的张量的形状为(frames_per_clip, H, W, C)，其中frames_per_clip表示每个序列中包含的帧数，H、W和C分别表示每个帧的高度、宽度和通道数。这样，我们就可以将每个采样窗口中的连续帧作为一个整体传递给模型进行处理。
        label_onehot = torch.zeros(2)
        label_onehot[label] = 1
        return np.float32(clip), np.float32(label_onehot)

        # sample = (sequence,label_onehot.long())
        # return sample  # 返回跌倒序列和跌倒标签'''
'''
def process_data(video_path, txt_path, frames_per_clip, step_between_clips, input_size):
    clips = []
    labels = []
    for filename in os.listdir(video_path):
        if filename.endswith(".avi"):
            video_file = os.path.join(video_path, filename)
            annotation_file = os.path.join(txt_path, f"{os.path.splitext(filename)[0]}.txt")
            start_frame, end_frame = read_txt(annotation_file)
            frames = load_video(video_file, input_size)
            for i in range(start_frame, end_frame, step_between_clips):
                if i + frames_per_clip >= num_frames:
                    continue
                clip = frames[i:i + frames_per_clip]
                clip = np.array(clip)
                clip = torch.from_numpy(clip).float()
                clips.append(clip)
                labels.append(1)
    clips = torch.stack(clips)
    labels = torch.tensor(labels)
    clips = clips.view(-1, frames_per_clip, input_size)
    return clips, labels
 '''
'''
def process_data(video_path, txt_path, frames_per_clip, step_between_clips):
    for filename in os.listdir(video_path):
        if filename.endswith(".avi"):
            video_file = os.path.join(video_path, filename)
            annotation_file = os.path.join(txt_path, f"{os.path.splitext(filename)[0]}.txt")
            start_frame, end_frame = read_txt(annotation_file)
            frames, num_frames = load_video(video_file)
            clips = []
            for i in range(start_frame, end_frame, step_between_clips):
                if i + frames_per_clip >= num_frames:
                    continue
                clip = frames[i:i + frames_per_clip]
                clip = np.array(clip)
                clip = clip.transpose(3, 0, 1, 2)  # 重新排列数组的维度，使通道维度成为第一个维度
                clip = torch.from_numpy(clip).float()
                clip = clip.permute(1, 0, 2, 3)  # (W, N, H, C) -> (N, W, H, C)帧维度是第一维度，通道维度是最后一个维度
                clip = clip.reshape(frames_per_clip, -1)  # (N, W, H, C) -> (N, W*H*C)展平张量，使其第二维（表示宽度维度）与第三维和第四维（表示高度和通道维度）组合在一起
                clips.append(clip)
            labels = [1] * len(clips)  # 初始化一个调用的新列表，其长度等于列表中的剪辑数，其中列表中的每个元素都设置为整数值 1
            yield torch.stack(clips), torch.tensor(labels)
            '''

'''
def process_data(video_path, txt_path, frames_per_clip, step_between_clips):# 都是差一步具体路
    X = []  # 存储一个个视频序列
    y = []  # 存储序列的类别
    for filename in os.listdir(video_path):
        if filename.endswith(".avi"):
            video_file = os.path.join(video_path, filename) #具体路已得
            annotation_file = os.path.join(txt_path, filename.replace(".avi", ".txt")) #具体路已得
            frames, num_frames = load_video(video_file) # 视频所有帧
            start_frame, end_frame = read_txt(annotation_file)
            #从start_frame开始，每隔step_between_clips帧采样一次，直到到达跌倒结束的end_frame - frames_per_clip帧为止。每次采样frames_per_clip帧并将其存储在X中，同时将对应的标签1存储在y中。这样就可以生成很多带有标签的视频剪辑，用于训练和测试模型。
            # 计算视频片段的数量
           # num_clips = int((num_frames - frames_per_clip) / step_between_clips + 1)

            if start_frame==0:
                for i in range(0, num_frames - frames_per_clip, step_between_clips):
                    X.append(frames[i:i+frames_per_clip])  # 将当前窗口内连续的帧序列添加到 X 列表中，一个位置存一个序列
                    y.append(0)  # 一个序列对应一个标签1，表示摔倒
            else:
                for i in range(start_frame, end_frame - frames_per_clip, step_between_clips):
                    X.append(frames[i:i+frames_per_clip])  # 将当前窗口内连续的帧序列添加到 X 列表中，一个位置存一个序列
                    y.append(1)  # 一个序列对应一个标签1，表示摔倒
                for i in range(0, start_frame - frames_per_clip, step_between_clips):
                        X.append(frames[i:i+frames_per_clip])  # 将当前窗口内连续的帧序列添加到 X 列表中，一个位置存一个序列
                        y.append(0)  # 一个序列对应一个标签0，表示没摔倒
                for i in range(end_frame,  num_frames - frames_per_clip, step_between_clips):
                    X.append(frames[i:i+frames_per_clip])  # 将当前窗口内连续的帧序列添加到 X 列表中，一个位置存一个序列
                    y.append(0)  # 一个序列对应一个标签0，表示没摔倒
    X = np.array(X) # 最终，所有视频所有分成的序列都在x里面了
    y = np.array(y)
    return X, y
'''

