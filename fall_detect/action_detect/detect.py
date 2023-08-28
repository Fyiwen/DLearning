import numpy as np
from torch import from_numpy, argmax
DEVICE = "cpu"

# net是一个分类网络，pose里面包含了一些整合起来的前面得到的姿态相关的信息集合，crownproportion是一个权衡值
def action_detect(net,pose,crown_proportion):  # 用于通过骨架图片识别出摔倒动作
    # img = cv2.cvtColor(pose.img_pose,cv2.IMREAD_GRAYSCALE)
    #maxHeight = pose.keypoints.max()  # 关节的最大高度
    #minHeight = pose.keypoints.min()  # 关节的最低高度
   # 以下的img操作是为了传入分类网络做准备
    img = pose.img_pose.reshape(-1)  # 将姿势图片的信息压缩成一行
    img = img / 255  # 进行数据归一化，把图片数据转成[0,1]之间的数据
    img = np.float32(img)  # 强制类型转换
    img = from_numpy(img[None,:]).cpu()  # 把数组转换成张量，只转图片的列信息
    predect = net(img)  # 把图像输入一个用来分类的网络，得到一个预测结果向量，这个网络较简单是net.py中的netv2

    action_id = int(argmax(predect,dim=1).cpu().detach().item())  # 取向量中数最大的那一个的下标，要么0要么1作为预测到的类别，这里已经得到了分类结果

    possible_rate = 0.6*predect[:,action_id] + 0.4*(crown_proportion-1) # 除了使用上面的网络输出得分外，还采取了人体框的宽高比来判断人有没有摔倒。这行就是在做权衡，这里0.6还是0.4是超参可改
    possible_rate = possible_rate.detach().numpy()[0]  # 把数值从计算图中分离出来，得到摔倒的可能性
    if possible_rate > 0.55:  # 如果这个值大于0.55就判断是摔倒姿态，0.55也是超参可以改
   # if maxHeight-minHeight < 50:
        pose.pose_action = 'fall'  #设置这个变量对应字符串fall
        if possible_rate > 1:
            possible_rate = 1  # 把大于1的可能性置1，是运算合理
        pose.action_fall = possible_rate  # 算出摔倒的具体可能性
        pose.action_normal = 1-possible_rate  # 算出正常姿态的具体可能性
    else:  # 如果小于等于就判断是正常姿态
        pose.pose_action = 'normal'
        if possible_rate >= 0.5:
            pose.action_fall = 1-possible_rate
            pose.action_normal = possible_rate
        else:
            pose.action_fall = possible_rate
            pose.action_normal = 1 - possible_rate

    return pose