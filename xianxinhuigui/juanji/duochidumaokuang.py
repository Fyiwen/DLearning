import torch
from d2l import torch as d2l
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

img = d2l.plt.imread('D:/pycode/xianxinhuigui/data/img/catdog.jpg')
h, w = img.shape[:2]

#  在特征图（fmap）上以每个像素作为锚框的中心生成锚框（anchors）。由于锚框中的xy轴坐标值（anchors）已经被除以特征图（fmap）的宽度和高度，因此这些值介于0和1之间，表示特征图中锚框的相对位置
# 给定特征图的宽度和高度fmap_w和fmap_h，以下函数将均匀地对任何输入图像中fmap_h行和fmap_w列中的像素进行采样。 以这些均匀采样的像素为中心，将会生成大小为s（假设列表s的长度为1）且宽高比（ratios）不同的锚框
# 比如特征图4*4，那么就会将原图均匀分成4*4部分，每部分中有一个中心像素，用以形成多个锚框
def display_anchors(fmap_w, fmap_h, s):
    d2l.set_figsize()
    # 前两个维度上的值不影响输出
    fmap = torch.zeros((1, 10, fmap_h, fmap_w)) # 初始化了一个特征图的形状，是虚假的特征图
    anchors = d2l.multibox_prior(fmap, sizes=s, ratios=[1, 2, 0.5]) #生成以特征图的每一个像素为中心的锚框们，形状为（1，fmaph*fmap_w*3，4）4是坐标信息，中间是锚框总个数一个中心3个框
    bbox_scale = torch.tensor((w, h, w, h)) # 原图的尺寸
    d2l.show_bboxes(d2l.plt.imshow(img).axes,
                    anchors[0] * bbox_scale) # 把按照特征图生成的锚框还原到原图上显示出来

display_anchors(fmap_w=4, fmap_h=4, s=[0.15]) # 可以看到，图像上4行和4列上的锚框的中心是均匀分布的。s代表锚框尺寸
display_anchors(fmap_w=2, fmap_h=2, s=[0.4]) # 特征图尺寸变小，锚框尺度变大
display_anchors(fmap_w=1, fmap_h=1, s=[0.8])
d2l.plt.show()