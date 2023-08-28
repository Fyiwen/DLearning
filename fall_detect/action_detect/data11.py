import numpy as np
import os
import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import argparse
import cv2
from torch import from_numpy, jit
from openpose_modules.keypoints import extract_keypoints, group_keypoints
from openpose_modules.pose import Pose
import os
from math import ceil, floor
from utils.contrastImg import coincide
import time
from runOpenpose import infer_fast
os.environ["PYTORCH_JIT"] = "0"
# openpose+lstm的处理
def run_demo(net, img, height_size, cpu, boxList):
    net = net.eval()  # 在测试环境下
    if not cpu:
        net = net.cuda()

    stride = 8  # 步幅
    upsample_ratio = 4  # 上采样率，用于放大图片
    num_keypoints = Pose.num_kpts  # 18，关节点个数

    heatmaps, pafs, scale, pad = infer_fast(net, img, height_size, stride, upsample_ratio,
                                                    cpu)  # 使用openpose模型网络，通过这个函数返回最终所得热图,paf,输入模型图象相比原始图像缩放倍数,输入模型图像的padding尺寸

    total_keypoints_num = 0  # 总共找到的关节点个数
    all_keypoints_by_type = []  # all_keypoints_by_type为18个list，每个list包含Ni个当前点的x、y坐标，当前点热图值，当前点在所有特征点中的index
    for kpt_idx in range(num_keypoints):  # 按照关节点个数进行循环，是19个，其中第19个为背景，前18个关节点
        total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type,
                                                         total_keypoints_num)  # 得到第i个关节点的个数,all_keypoints_by_type中是关节点的[x,y,conf,id]

    pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs,
                                                          demo=True)  # 得到所有分配的人的信息，及所有关节点信息
    for kpt_id in range(all_keypoints.shape[0]):  # 遍历每一个检测到的关节点，依次将每个关节点信息缩放回原始图像上
        all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
        all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
    current_poses = []
    for n in range(len(pose_entries)):  # 依次遍历找到的每个人
        if len(pose_entries[n]) == 0:
            continue
        pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
        for kpt_id in range(num_keypoints): # 遍历每一个关节点类型
            if pose_entries[n][kpt_id] != -1.0:  # 如果这个人的信息中，此关节点类型索引值对应的维度值不等于-1，则表示这个人已经被被分配到了这个类型的关节点
                pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])  # 记录这个关节的x坐标
                pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])  # 记录这个关节的y坐标
        pose = Pose(pose_keypoints, pose_entries[n][18])  # pose_entries[n][18]每个姿势的得分对应Pose这个类中传入的confidence
        posebox = (int(pose.bbox[0]), int(pose.bbox[1]), int(pose.bbox[0]) + int(pose.bbox[2]),
                           int(pose.bbox[1]) + int(pose.bbox[3]))  # （框左上角x，左上角y，右下角x，右下角y）
        if boxList:#从这儿
            coincideValue = coincide(boxList, posebox)
            print(posebox)
            print('coincideValue:' + str(coincideValue))
            if len(pose.getKeyPoints()) >= 10 and coincideValue >= 0.3 and pose.lowerHalfFlag < 3:
                current_poses.append(pose)
        else:
            current_poses.append(pose)
    for pose in current_poses:
        pose.img_pose = pose.draw(img, is_save=True, show_draw=True)  # 画出这个姿势的骨骼图片
        return pose.img_pose






# 要使用openpose的话，不仅要加上上面这个函数，而且对帧的处理也要改，因为这个数据集里的视频有没人的情况，或者有人但是openpose测不出来，那么就生不成骨架图片，那么就没有返回的骨骼图片就会报错,而且形状为空，最后整合成序列也会报错。但是代码逻辑我不想改，所以选择筛选好的视频来作为数据集。
#发现筛不了，手里的视频都有问题，所以增加了一段判断逻辑
# 而且不同于train1，他现在经过openpose之后不再是三通道图片，而是1通道的黑白图片，所以代码形状上要改一改。
def load_video(video_file, input_size):
    cap = cv2.VideoCapture(video_file)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    net = jit.load(r'F:\bishe\fall_detect\action_detect\checkPoint\openpose.jit')
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (input_size, input_size))
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #frame = frame / 255.0
        frame = run_demo(net, frame, input_size, True, [])
        if frame is None:
            frame = np.zeros((32, 32), dtype=np.float32)
        else:
            pass
        #frame = torch.from_numpy(frame)  # numpy.ndarray类型的数据转换为tensor才能用permute,现在形状（input_size，input_size)
        frame=np.expand_dims(frame, axis=0)  # （1,input_size，input_size)
        #frame = frame.permute(2, 0, 1)  # (3, input_size, input_size)
        #frame = frame.numpy()  # 再转回去
        frame = np.expand_dims(frame, axis=0)  # numpy用这个(1, 1, input_size, input_size)
        print(frame.shape)
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
            frames, num_frames = load_video(video_file, input_size) # (num_frames, 1, input_size, input_size)
            start_frame, end_frame = read_txt(annotation_file)
            clips = []
            if start_frame == 0:
                for i in range(0, num_frames - frames_per_clip, step_between_clips):
                    clip = np.array(frames[i:i+frames_per_clip]) # (frames_per_clip, 1, input_size, input_size)

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
