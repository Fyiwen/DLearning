#边缘框实现
import torch
from d2l import torch as d2l
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

d2l.set_figsize()
img = d2l.plt.imread('D:/pycode/xianxinhuigui/data/img/catdog.png')
d2l.plt.imshow(img)

# 定义在这两种表示法之间进行转换的函数：box_corner_to_center从两角表示法转换为中心宽度表示法，而box_center_to_corner反之亦然。 输入参数boxes可以是长度为4的张量，也可以是形状为（n，4）的二维张量，其中n是边界框的数量。
def box_corner_to_center(boxes):
    """从（左上，右下）转换到（中间，宽度，高度）"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = torch.stack((cx, cy, w, h), axis=-1)  # ？
    return boxes

#@save
def box_center_to_corner(boxes):
    """从（中间，宽度，高度）转换到（左上，右下）"""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h  # 显示器图像的坐标系是左上角为原点，x方向向右为正，y方向向下为正,所以用-
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = torch.stack((x1, y1, x2, y2), axis=-1)
    return boxes
# 根据坐标信息定义图像中狗和猫的边界框。 图像中坐标的原点是图像的左上角，向右的方向为x轴的正方向，向下的方向为y轴的正方向
dog_bbox, cat_bbox = [23.0, 21.0, 211.0, 215.0], [178.0, 60.0, 317.0, 214.0]

#转换两次来验证边界框转换函数的正确性
#boxes = torch.tensor((dog_bbox, cat_bbox))
#box_center_to_corner(box_corner_to_center(boxes)) == boxes


#可以将边界框在图中画出，以检查其是否准确。 画之前，我们定义一个辅助函数bbox_to_rect。 它将边界框表示成matplotlib的边界框格式
def bbox_to_rect(bbox, color):
    # 将边界框(左上x,左上y,右下x,右下y)格式转换成matplotlib格式：
    # ((左上x,左上y),宽,高)
    return d2l.plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=2)
# 在图像上添加边界框之后，我们可以看到两个物体的主要轮廓基本上在两个框内
fig = d2l.plt.imshow(img)
fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'))
d2l.plt.show() # 这句话很重要，有了图才能出来






