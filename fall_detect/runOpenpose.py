import argparse
import cv2
import numpy as np
from torch import from_numpy, jit
from openpose_modules.keypoints import extract_keypoints, group_keypoints
from openpose_modules.pose import Pose
from action_detect.detect import action_detect
import os
from math import ceil, floor
from utils.contrastImg import coincide
import time
os.environ["PYTORCH_JIT"] = "0"


class ImageReader(object):  # 用于读取图像
    def __init__(self, file_names):
        self.file_names = file_names
        self.max_idx = len(file_names)  # 文件名的长度

    def __iter__(self):  # 返回迭代器自身
        self.idx = 0
        return self

    def __next__(self):  # 返回容器中的下一个值
        if self.idx == self.max_idx:
            raise StopIteration  # 完成上面指定的判断，触发StopIteration异常结束迭代
        img = cv2.imread(self.file_names[self.idx], cv2.IMREAD_COLOR)  # 以彩色图的形式读取图片
        if img.size == 0:
            raise IOError('Image {} cannot be read'.format(self.file_names[self.idx]))  # 触发无法读取图片的异常
        self.idx = self.idx + 1
        return img


class VideoReader(object):  # 用于读取视频
    def __init__(self, file_name, code_name):
        self.file_name = file_name
        self.code_name = str(code_name)  # 文字内容
        try:  # OpenCV needs int to read from webcam
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)  # 生成读取视频或摄像头的对象

        if not self.cap.isOpened():  # 判断有没有读取成功
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()  # 按帧读取视频，was_read是布尔型，正确读取则返回True，读取失败或读取视频结尾则会返回False,img为每一帧的图像
        if not was_read:
            raise StopIteration

        # print(self.cap.get(7),self.cap.get(5))
        cv2.putText(img, self.code_name, (5, 35),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))  # 将文字添加到图片上的这个位置，以及字体类型，大小，颜色，粗细没写
        return img


def normalize(img, img_mean, img_scale):  # 实现图片归一化，图片去均值，就是为了特征数据标准化
    img = np.array(img, dtype=np.float32)  # 把图片转成数组
    img = (img - img_mean) * img_scale
    return img


def pad_width(img, stride, pad_value, min_dims):
    h, w, _ = img.shape  # 收集图片的宽高信息，这里传入的是缩放后的图片
    h = min(min_dims[0], h)  # 选择小的内个作为高，min_dims[0]网络期望的输入高度
    min_dims[0] = ceil(min_dims[0] / float(stride)) * stride  # 向上取整
    min_dims[1] = max(min_dims[1], w)  #  min_dims[1]=max(期望高，图像缩放后宽度）
    min_dims[1] = ceil(min_dims[1] / float(stride)) * stride  # 向下取整
    pad = []
    pad.append(int(floor((min_dims[0] - h) / 2.0)))  # 得到pad[0]
    pad.append(int(floor((min_dims[1] - w) / 2.0)))  # 得到pad[1]
    pad.append(int(min_dims[0] - h - pad[0]))  # 得到pad[2]
    pad.append(int(min_dims[1] - w - pad[1]))  # 得到pad[3]
    padded_img = cv2.copyMakeBorder(img, pad[0], pad[2], pad[1], pad[3],
                                    cv2.BORDER_CONSTANT, value=pad_value)  # 使用拓展边界函数进行图片填充便于处理边界，pad[0]是在图片上边界向上拓展的行数，pad[2]在图片下边界向上拓展的行数，pad[1]在图片左边界向上拓展的行数，pad[3]在图片右边界向上拓展的行数
    return padded_img, pad  # 返回填充后的图片和这个四维的pad


def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=(128, 128, 128), img_scale=1 / 256):
    height, width, _ = img.shape  # 得到实际高宽
    scale = net_input_height_size / height  # 得到可以将图片实际高缩放到期望高的缩放倍数，图片期望高则是网络所要求的输入图片大小

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)  # 得到缩放后的图像，这里dsize被设置为0（None），则按fx与fy与原始图像大小相乘得到输出图像尺寸大小，这样写比直接给尺寸好
    scaled_img = normalize(scaled_img, img_mean, img_scale)  # 归一化图像，为了标准
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]  # 这个min_dim参数由网络期望的输入图片高和max(期望高，图像缩放后宽度）组成
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)  # 填充到高宽为stride整数倍的值，得到填充后的图片和填充信息

    tensor_img = from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()  # 把HWC转成CHW(BGR格式)适应网络输入，填充后图片的维度转换。permute函数将tensor的维度进行相应转换，原本的tensor维度为（1,3,3）现在变成了（3，1,3），然后通过unsqueeze函数又在0层创造了一维，所以现在的tensor从3维变成4维。
    if not cpu:
        tensor_img = tensor_img.cuda()

    stages_output = net(tensor_img)  # 转换维度后的图片送入网络得到网络输出，这个网络经过我层层找发现是openpose.jit文件里，怀疑是openpose里用的那个网络。所以这个输出应该是包含最后一个阶段的热图和paf图

    # print(stages_output)

    stage2_heatmaps = stages_output[-2]  # 得到最后一个stage的热图
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))  # 将最后一个stage的热图作为最终的热图，这里是把图片的维度格式换一下又转回HWC（因为图片参数的格式可能不符合它会被输入进的函数对应参数所需要的格式）
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio,
                          interpolation=cv2.INTER_CUBIC)  # 将热图放大upsample_ratio倍

    stage2_pafs = stages_output[-1]  # 得到最后一个stage的paf图
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))  # 转换维度后将最后一个stage的paf作为最终的paf
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio,
                      interpolation=cv2.INTER_CUBIC)  # 将paf将放大upsample_ratio倍

    return heatmaps, pafs, scale, pad  # 返回最后一阶段得到的热图,paf,输入模型图像相比原始图像缩放倍数,输入模型图像的padding尺寸


def run_demo(net, action_net, image_provider, height_size, cpu, boxList):
    net = net.eval()  # 在测试环境下
    if not cpu:
        net = net.cuda()

    stride = 8  # 步幅
    upsample_ratio = 4  # 上采样率，用于放大图片
    num_keypoints = Pose.num_kpts  # 18，关节点个数

    i = 0
    for img in image_provider:  # 遍历图像集，拿出每一张图像
        orig_img = img.copy()  # copy 一份图片
         # print(i)
        fallFlag = 0
        if i % 1 == 0:
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
                    coincideValue = coincide(boxList, posebox)  # 算yolo的框和据姿态得出的框的重合比例
                    print(posebox)  # 输出框的信息
                    print('coincideValue:' + str(coincideValue))  #输出
                    if len(pose.getKeyPoints()) >= 10 and coincideValue >= 0.3 and pose.lowerHalfFlag < 3:  # 当人体的点数大于10个的时候算作一个人,同时判断yolov5的框和pose的框是否有交集并且占比30%,同时要有下半身
                        current_poses.append(pose)
                else:# 到这可以不要，没用yolo
                    current_poses.append(pose)
            for pose in current_poses:
                pose.img_pose = pose.draw(img, is_save=True, show_draw=True)  # 画出这个姿势的骨骼图片
                crown_proportion = pose.bbox[2] / pose.bbox[3]  # 姿态框的宽高比
                pose = action_detect(action_net, pose, crown_proportion)  # 把每个检测出来的姿态送入判断，此姿态摔倒还是正常

                if pose.pose_action == 'fall':# 如果结果是摔倒，把这个姿态框出来，并且框上放显示状态的字
                    cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                                  (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 0, 255), thickness=3)
                    cv2.putText(img, 'state: {}'.format(pose.pose_action), (pose.bbox[0], pose.bbox[1] - 16),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
                    fallFlag = 1
                else:  # 结果不是摔倒也框出来并显示状态
                    cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                                  (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
                    cv2.putText(img, 'state: {}'.format(pose.pose_action), (pose.bbox[0], pose.bbox[1] - 16),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0))
                    # fallFlag = 1
            # if fallFlag == 1:
            #     t = time.time()
            #     cv2.imwrite(f'C:/zqr/project/yolov5_openpose/Image/{t}.jpg', img)
            #     print('我保存照片了')

            img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)  # 把原图和画有所有标记的图片叠加，直观来看让处理后的那些框啥的变淡了只是
            # 保存识别后的照片
            t=time.time()  # 新增
            cv2.imwrite(f'F:/bishe/fall_detect/shibie_result/{t}.jpg', img) # 这里本来注释掉了，为了detect摄像头抓拍可以停下而取消buduibudui
            cv2.imshow('Lightweight Human Pose Estimation Python Demo', img)  # 以前面的名字显示图像，cv2.imshow之后要跟cv2.waitkey

            cv2.waitKey(10)# 等待键盘输入，实际上，waitkey控制着imshow的持续时间
        i += 1
        # print(i)
        #if(i>2):
        #    break
    cv2.destroyAllWindows()  # 轻易删除任何我们建立的所有窗口


def detect_main(video_name='',video_source = '',image_source = ''):##可视化时加上video_source = '',image_source = '',
    parser = argparse.ArgumentParser(   # 创建一个 ArgumentParser 对象。ArgumentParser 对象包含将命令行解析成 Python 数据类型所需的全部信息
        description='''Lightweight human pose estimation python demo.
                           This is just for quick results preview.
                           Please, consider c++ demo for the best performance.''')  # 给一个 ArgumentParser 添加程序参数信息
    parser.add_argument('--checkpoint-path', type=str, default='openpose.jit',
                        help='path to the checkpoint')
    parser.add_argument('--height-size', type=int, default=256, help='network input layer height size')
    parser.add_argument('--video', type=str, default='F:\\bishe\\fall_detect\\data\\video\\2.mp4', help='path to video file or camera id')
    # 用下面这个检测图片，上面这个检测视频,注意视频时下行直接注释掉可以，但是图片时上一行不能直接注释而是应把default里面删掉F:\\bishe\\fall_detect\\data\\video\\2.mp4，，F:\\bishe\\fall_detect\\data\\Lei2\\Home_02\\Videos\\video (48).avi
    # 而且可视化的时候，也是要对应，比如要检测视频那么要把下行注释
   # parser.add_argument('--images', nargs='+',
    #                   default='F:\\bishe\\fall_detect\\data\\pics\\8.png',  # 训练的时候路径改成F:\\bishe\\fall_detect\\data\\FallDataset\\img,F:\\bishe\\fall_detect\\data\\pics
    #                   help='path to input image(s)')
    parser.add_argument('--cpu', action='store_true', help='run network inference on cpu')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    parser.add_argument('--code_name', type=str, default='None', help='the name of video')
    # parser.add_argument('--track', type=int, default=0, help='track pose id in video')
    # parser.add_argument('--smooth', type=int, default=1, help='smooth pose keypoints')
    args = parser.parse_args()  # 解析参数
    args.video = video_source##检测视频可视化的时候加上
    args.images = image_source##检测图片可视化的时候加上
    if video_name != '':  # 如果视频名称不为空
        args.code_name = video_name

    if args.video == '' and args.images == '':  # 如果找不到视频和图像则返回异常
        raise ValueError('Either --video or --image has to be provided')

    net = jit.load(r'.\action_detect\checkPoint\openpose.jit')  # 载入这个模型是可以识别出人体姿态的网络

    # *************************************************************************
    action_net = jit.load(r'.\action_detect\checkPoint\action.jit')  # 载入这个模型作为可以识别出摔倒动作的网络
    # ************************************************************************
    # frame_provider = ImageReader(args.images) ##
    if args.video != '':
        frame_provider = VideoReader(args.video, args.code_name)  # 读取视频
    else:
        images_dir = []
        if os.path.isdir(args.images):  # 判断这个路径是不是目录
            for img_dir in os.listdir(args.images):  # 返回指定的文件夹包含的文件或文件夹的名字的列表
                images_dir.append(os.path.join(args.images, img_dir))  # 将路径名连接起来，并且将得到的结果都放在images_dir中
            frame_provider = ImageReader(images_dir)  # 读取图片
        else:
            img = cv2.imread(args.images, cv2.IMREAD_COLOR)  # 从路径中读取彩色图片
            frame_provider = [img]

        # *************************************************************************

        # args.track = 0
    # camera = VideoReader('rtsp://admin:a1234567@10.34.131.154/cam/realmonitor?channel=1&subtype=0',args.code_name)

    run_demo(net, action_net, frame_provider, args.height_size, True, [])


if __name__ == '__main__':
    detect_main()
