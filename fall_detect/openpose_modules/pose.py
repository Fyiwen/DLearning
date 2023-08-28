import cv2
import numpy as np

from openpose_modules.keypoints import BODY_PARTS_KPT_IDS, BODY_PARTS_PAF_IDS
from openpose_modules.one_euro_filter import OneEuroFilter
import time


class Pose:
    num_kpts = 18  # 关节点总数
    kpt_names = ['nose', 'neck',
                 'r_sho', 'r_elb', 'r_wri', 'l_sho', 'l_elb', 'l_wri',
                 'r_hip', 'r_knee', 'r_ank', 'l_hip', 'l_knee', 'l_ank',
                 'r_eye', 'l_eye',
                 'r_ear', 'l_ear']  # 关节点名称
    sigmas = np.array([.26, .79, .79, .72, .62, .79, .72, .62, 1.07, .87, .89, 1.07, .87, .89, .25, .25, .35, .35],
                      dtype=np.float32) / 10.0
    vars = (sigmas * 2) ** 2
    last_id = -1
    color = [0, 224, 255]

    def __init__(self, keypoints, confidence):  # 初始化
        super().__init__()
        self.keypoints = keypoints
        self.confidence = confidence
        self.bbox = Pose.get_bbox(self.keypoints)
        self.pose_action = None
        self.action_fall = None
        self.action_normal = None
        self.img_pose = None
        self.id = None
        self.filters = [[OneEuroFilter(), OneEuroFilter()] for _ in range(Pose.num_kpts)]
        self.lowerHalfFlag = lowerHalf(self.keypoints)


    @staticmethod  # 封装成静态方法，类中的某个方法既不需要访问实例属性或者调用实例方法，同时也不需要访问类属性或者调用类方法
    def get_bbox(keypoints):  # 此方法用于根据一个人检测出的所有关节点，得到一个可以包住这个人的框
        found_keypoints = np.zeros((np.count_nonzero(keypoints[:, 0] != -1), 2), dtype=np.int32) # 初始化，最终存就是一个人的轮廓点集合
        found_kpt_id = 0
        for kpt_id in range(Pose.num_kpts):  # 遍历每一个关节类型
            if keypoints[kpt_id, 0] == -1:  # 这个关节类型没找到对应关节
                continue
            found_keypoints[found_kpt_id] = keypoints[kpt_id]  # 这个关节类型的关节的信息存入
            found_kpt_id += 1
        bbox = cv2.boundingRect(found_keypoints)  # 得到一个边框，可以把这个人包起来
        return bbox  # 返回框  返回四个值，分别是x，y，w，h；x，y是矩阵左上点的坐标，w，h是矩阵的宽和高

    # def update_id(self, id=None):
    #     self.id = id
    #     if self.id is None:
    #         self.id = Pose.last_id + 1
    #         Pose.last_id += 1

    def getKeyPoints(self):

        assert self.keypoints.shape == (Pose.num_kpts, 2)
        points = []
        for part_id in range(len(BODY_PARTS_PAF_IDS) - 2):
            kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]  # 躯干的终点
            global_kpt_b_id = self.keypoints[kpt_b_id, 0]  # 此终点关节点在所有关节点中的索引
            if global_kpt_b_id != -1:  # 终点检测到了
                # x_b, y_b = self.keypoints[kpt_b_id]
                points.append(self.keypoints[kpt_b_id])  # 终点的关节信息都存这里

        gcn_points = np.array(points)

        # for point in gcn_points[:13]:
        #     x_b, y_b = point
        #     cv2.circle(img, (int(x_b), int(y_b)), 3, Pose.color, -1)

        return gcn_points[:13]  # 返回终点关节的个数

    def draw(self, img,is_save = False,show_draw = True):  # 此方法画出骨骼图片
        assert self.keypoints.shape == (Pose.num_kpts, 2)

        # **************************************************
        iw, ih = self.bbox[2],self.bbox[3]  # 这是框的宽高，框里是某一个人
        w, h = 128, 128  # 这是真实图片的大小
        I = np.zeros((128, 128), dtype=np.uint8)  # 一个真实图片大小的空白画布，用来画此人的骨骼图片
        # I = cv2.cvtColor(I, cv2.COLOR_GRAY2BGR)
        if iw == 0 or ih == 0 or w == 0 or h == 0:
            print("erro,width and height == 0")
        else:
            scale = min(float(w) / float(iw), float(h) / float(ih))  # 转换的最小比例
        # **************************************************

        for part_id in range(len(BODY_PARTS_PAF_IDS) - 2):   # 遍历每一个躯干
            kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]   # 当前躯干起点的id
            global_kpt_a_id = self.keypoints[kpt_a_id, 0]  # 此起点关节点在所有关节点中的索引
            if global_kpt_a_id != -1:  # 起点存在，即被分配了当前关节点
                x_a, y_a = self.keypoints[kpt_a_id]  # 当前起点在原图像上的坐标
                x_a, y_a = int(x_a),int(y_a)  # 转成int型
                if show_draw:
                    cv2.circle(img, (x_a, y_a), 3, Pose.color, -1)   # 在原图，以此关节位置为中心，3为半径，绘制实心圆，即在图上画出关节圆点

            # **************************************
                px_a, py_a = x_a-self.bbox[0], y_a-self.bbox[1]  # bbox[0],[1]是框左上角的xy坐标

                cv2.circle(I, (int(px_a*scale), int(py_a*scale)),3, [255, 255, 255] , -1) # 在空白画布上对应比例位置，也画出这个关节点圆，实心的
            # **************************************

            kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]   # 当前躯干终点的id
            global_kpt_b_id = self.keypoints[kpt_b_id, 0]    # 当前终点在所有关节点中的索引
            if global_kpt_b_id != -1:
                x_b, y_b = self.keypoints[kpt_b_id]
                x_b, y_b = int(x_b),int(y_b)
                if show_draw:
                    cv2.circle(img, (x_b, y_b), 3, Pose.color, -1)
                # **************************************
                px_b, py_b = x_b - self.bbox[0], y_b - self.bbox[1]  # bbox[0],[1]是框左上角的xy坐标

                cv2.circle(I, (int(px_b * scale), int(py_b * scale)), 3,[255, 255, 255], -1)  # 在空白画布上对应比例位置，也画出这个关节点圆
            #**************************************
            if global_kpt_a_id != -1 and global_kpt_b_id != -1:  # 若此躯干的起点和终点均被分配
                if show_draw:
                    cv2.line(img, (x_a, y_a), (x_b, y_b), Pose.color, 2)  # 在原图，画连接起点和终点的直线
                # **************************************

                cv2.line(I, (int(px_a*scale), int(py_a*scale)), (int(px_b*scale), int(py_b*scale)),[255, 255, 255], 2)  # 在I上画这根肢体线
            #**************************************
        # 保存画完的骨骼图片I
        if is_save:
            t = time.time()
            t = int(round(t * 1000))
            cv2.imwrite(f'F:/bishe/fall_detect/data/train/gugetu/{t}.jpg',I)
        # **************************************

        return I  # 最终I是一个人的骨架图片




def get_similarity(a, b, threshold=0.5): # 此方法用于判断两个姿势之间的关节点的相似度
    num_similar_kpt = 0
    for kpt_id in range(Pose.num_kpts): # 遍历每一个关节类型
        if a.keypoints[kpt_id, 0] != -1 and b.keypoints[kpt_id, 0] != -1: # 这两个人的这个关节类型都被分配了关节
            distance = np.sum((a.keypoints[kpt_id] - b.keypoints[kpt_id]) ** 2)  # 这两个关节之间在图上的距离
            area = max(a.bbox[2] * a.bbox[3], b.bbox[2] * b.bbox[3]) # 取a的框和b的框中最大的那个大小
            similarity = np.exp(-distance / (2 * (area + np.spacing(1)) * Pose.vars[kpt_id]))  # 相似度
            if similarity > threshold:  # 相似度超过阈值
                num_similar_kpt += 1
    return num_similar_kpt  # 返回相似关节个数


def track_poses(previous_poses, current_poses, threshold=3, smooth=False):
    """Propagate poses ids from previous frame results. Id is propagated,
    if there are at least `threshold` similar keypoints between pose from previous frame and current.
    If correspondence between pose on previous and current frame was established, pose keypoints are smoothed.
    :param previous_poses: poses from previous frame with ids
    :param current_poses: poses from current frame to assign ids
    :param threshold: minimal number of similar keypoints between poses
    :param smooth: smooth pose keypoints between frames
    :return: None
    """
    current_poses = sorted(current_poses, key=lambda pose: pose.confidence, reverse=True)  # 将当前帧的所有姿势按照置信度排序
    mask = np.ones(len(previous_poses), dtype=np.int32)
    for current_pose in current_poses:  # 遍历当前帧中所有的姿势
        best_matched_id = None
        best_matched_pose_id = None
        best_matched_iou = 0  # 最佳匹配的相似度，最终存的是当前帧中的这个姿势能与以前帧中的某个姿势所达到的最高关节相似度
        for id, previous_pose in enumerate(previous_poses):  # 遍历以前帧中的所有姿势
            if not mask[id]:
                continue
            iou = get_similarity(current_pose, previous_pose)  # 当前帧中的一个姿势与以前帧中的一个姿势的关节相似度
            if iou > best_matched_iou:  # 出现了更高的相似度
                best_matched_iou = iou  # 记下此暂时最高的相似度
                best_matched_pose_id = previous_pose.id  # 记下以前帧中这个姿势的id
                best_matched_id = id  # 记下这个以前帧的id，因为有很多的以前帧
        if best_matched_iou >= threshold:  # 最终的最佳相似度超过了阈值
            mask[best_matched_id] = 0   # 这个最佳匹配的以前帧，mask置0，后面不再用作比较。
        else:  # 当前帧中的这个姿势和以前帧中的都不相似
            best_matched_pose_id = None
        current_pose.update_id(best_matched_pose_id)

        # if smooth:
        #     for kpt_id in range(Pose.num_kpts):
        #         if current_pose.keypoints[kpt_id, 0] == -1:
        #             continue
        #         # reuse filter if previous pose has valid filter
        #         if (best_matched_pose_id is not None
        #                 and previous_poses[best_matched_id].keypoints[kpt_id, 0] != -1):
        #             current_pose.filters[kpt_id] = previous_poses[best_matched_id].filters[kpt_id]
        #         current_pose.keypoints[kpt_id, 0] = current_pose.filters[kpt_id][0](current_pose.keypoints[kpt_id, 0])
        #         current_pose.keypoints[kpt_id, 1] = current_pose.filters[kpt_id][1](current_pose.keypoints[kpt_id, 1])
        #     current_pose.bbox = Pose.get_bbox(current_pose.keypoints)
 # 检验有没有下半身 flag ==4 则没有膝盖往下
def lowerHalf(boxList):
    flag = 0
    for a in boxList[9:11]:
        if a[0] == -1:
            flag += 1
    for b in boxList[12:14]:
        if b[0] == -1:
            flag += 1
    return  flag