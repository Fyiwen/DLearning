# 处理Lei2数据集为了transformer
import numpy as np
import os
import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

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

        frame = torch.from_numpy(frame)  # numpy.ndarray类型的数据转换为tensor
        frame = frame.permute(2, 0, 1)  # (3, input_size, input_size)
        frame = frame.numpy()  # 再转回去
        frame = np.expand_dims(frame, axis=0)  # numpy用这个(1, 3, input_size, input_size)
        frames.append(frame)
    cap.release()
    frames = np.concatenate(frames, axis=0)  # 如果使用NumPy，可以使用np.concatenate函数将所有张量连接起来，(num_frames, 3, input_size, input_size)的四维张量。
    return frames, num_frames

def read_txt(txt_path):#具体路
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

                    yield clip, 0
            else:
                for i in range(start_frame, end_frame - frames_per_clip, step_between_clips):
                    clip = np.array(frames[i:i+frames_per_clip])

                    yield clip, 1
                for i in range(0, start_frame - frames_per_clip, step_between_clips):
                    clip = np.array(frames[i:i+frames_per_clip])

                    yield clip, 0


class FallDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, video_path, txt_path, frames_per_clip, step_between_clips, input_size):
        self.samples = list(process_data(video_path, txt_path, frames_per_clip, step_between_clips, input_size))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        clip, label = self.samples[index]
        clip = torch.from_numpy(clip).float()
        return clip, label

