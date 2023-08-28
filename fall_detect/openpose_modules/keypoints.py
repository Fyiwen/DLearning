import math
import numpy as np
from operator import itemgetter
# 每个数字对应表示一个关节，一组数字表示由两个关节连接起来的躯干
BODY_PARTS_KPT_IDS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11],
                      [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17], [2, 16], [5, 17]]
BODY_PARTS_PAF_IDS = ([12, 13], [20, 21], [14, 15], [16, 17], [22, 23], [24, 25], [0, 1], [2, 3], [4, 5],
                      [6, 7], [8, 9], [10, 11], [28, 29], [30, 31], [34, 35], [32, 33], [36, 37], [18, 19], [26, 27])
# 一共十九组表示躯干的paf图，因为paf中的单位矢量v=（x，y）不好表示，所以前一位是对应paf_x,后一位是对应paf_y

def linspace2d(start, stop, n=10):  # 此函数实现开始值为start，结束值为stop，他们的中间包括他们俩均匀分布10个点，即在两点之间按线性分布产生n个点的等差数组
    points = 1 / (n - 1) * (stop - start)
    return points[:, None] * np.arange(n) + start[:, None]


def extract_keypoints(heatmap, all_keypoints, total_keypoint_num):
    # 以下在找热图中的峰值点
    heatmap[heatmap < 0.1] = 0  # 热图中值小于0.1的地方都直接置0
    heatmap_with_borders = np.pad(heatmap, [(2, 2), (2, 2)], mode='constant')  # 给热图进行数值填充，各维度的各个方向上想要填补的长度为2，连续填充相同的值，即在图像四周边缘填充0，使得卷积运算后图像大小不会缩小，同时也不会丢失边缘和角落的信息
    heatmap_center = heatmap_with_borders[1:heatmap_with_borders.shape[0]-1, 1:heatmap_with_borders.shape[1]-1]  # 提取中心部分的热图，即去掉周围一圈
    heatmap_left = heatmap_with_borders[1:heatmap_with_borders.shape[0]-1, 2:heatmap_with_borders.shape[1]]  # 取右边的图，左边一圈切掉
    heatmap_right = heatmap_with_borders[1:heatmap_with_borders.shape[0]-1, 0:heatmap_with_borders.shape[1]-2]  # 取左边的图
    heatmap_up = heatmap_with_borders[2:heatmap_with_borders.shape[0], 1:heatmap_with_borders.shape[1]-1]  # 取下边的图
    heatmap_down = heatmap_with_borders[0:heatmap_with_borders.shape[0]-2, 1:heatmap_with_borders.shape[1]-1] # 取上边的图

    heatmap_peaks = (heatmap_center > heatmap_left) &\
                    (heatmap_center > heatmap_right) &\
                    (heatmap_center > heatmap_up) &\
                    (heatmap_center > heatmap_down)  # 找到热图的峰值区域，两个矩阵&操作，如果center点的值大于left,right,bottom,top点的值，那center的点就是峰值点。
    heatmap_peaks = heatmap_peaks[1:heatmap_center.shape[0]-1, 1:heatmap_center.shape[1]-1]  # 这里在切片，因为之前的heatmap是padding了周围一圈，要切掉
    keypoints = list(zip(np.nonzero(heatmap_peaks)[1], np.nonzero(heatmap_peaks)[0]))  # 得到数组中非零元素索引，即可能是关节的峰值的位置，打包成一个个元组，再将元组转换成列表，返回由这些元组组成的列表。np.nonzero(heatmap_peaks)[1]某个峰值点x坐标，【0】y坐标，一个列表中多个元组，一个元组表示一个峰值点（x，y），
    keypoints = sorted(keypoints, key=itemgetter(0))  # 将所有得到的关节点进行排序，itemgetter(0)用于获取对象的第0维的数据,key为函数，指定取待排序元素的哪一项进行排序。所以这里按照关节点第0维的数据进行排序
    # 到这儿为止
    suppressed = np.zeros(len(keypoints), np.uint8)  # 定义了一个全0的数组，行数是关节点个数，列是专门存储图像的数量
    keypoints_with_score_and_id = []  # 初始化
    keypoint_num = 0  # 统计，从热图中提取出的峰值中一共有多少真的关节
    for i in range(len(keypoints)):  # 针对每一个关节做的操作，判断第i个关节点是不是重复的，因为上面热图中得到了很多峰值点即猜测关节
        if suppressed[i]:  # suppressed[i]=1这里就跳出本次循环，因为1代表它已经被判断过并且是重复的节点所以不需要再判断
            continue
        for j in range(i+1, len(keypoints)):  # 通过ij的取值两两比较所有检测出来的关节，如果两个关节点位置靠近则认为是同一个关节
            if math.sqrt((keypoints[i][0] - keypoints[j][0]) ** 2 +
                         (keypoints[i][1] - keypoints[j][1]) ** 2) < 6:  # keypoints[i][0]是第i个关节点的x坐标，keypoints[i][1]第i个关节点的y坐标
                suppressed[j] = 1  # 因为位置靠近算是同一个就置1，但是这里都不比一下峰值哪个大就盲目舍去一个不合理
        keypoint_with_score_and_id = (keypoints[i][0], keypoints[i][1], heatmap[keypoints[i][1], keypoints[i][0]],  # heatmap里面似乎是以（y，x）坐标定位
                                      total_keypoint_num + keypoint_num)  # 这个参数里包含，第i个关节的xy坐标，这个坐标位置的热图得分和当前这个关节点是我统计的第多少个关节点
        keypoints_with_score_and_id.append(keypoint_with_score_and_id) # 把统计到的叠加放在一起
        keypoint_num += 1  # 关节数量增加
    all_keypoints.append(keypoints_with_score_and_id)  # 一次次随着循环加进去，添加提取的所有的关节点的信息。如果all_keypoints[0]里面是所有属于0这个关节类型的检测到的所有关节点的信息，即它们每个人的关节点坐标啊得分啊那些
    return keypoint_num  # 返回确定的提取到的真实关节点数


def group_keypoints(all_keypoints_by_type, pafs, pose_entry_size=20, min_paf_score=0.05, demo=False):
    pose_entries = []  # 初始化
    all_keypoints = np.array([item for sublist in all_keypoints_by_type for item in sublist])  # 将列表转成数组 ，将所有关节点展开成数组，可以debug看一下
    for part_id in range(len(BODY_PARTS_PAF_IDS)):  # 循环遍历每一个躯干
        part_pafs = pafs[:, :, BODY_PARTS_PAF_IDS[part_id]]  # 从网络输出的38个paf图中提取出属于当前躯干的两张paf图（根据当前躯干编号找到BODY_PARTS_PAF_IDS中对应的一组[]里面存了paf_x,paf_y的编号，表示第几张paf图存了这个躯干的paf_x，第几张存了这个paf_y

        kpts_a = all_keypoints_by_type[BODY_PARTS_KPT_IDS[part_id][0]]  # 当前躯干的起点关节的类型找到了，属于这个类型的所有检测到的关节点
        kpts_b = all_keypoints_by_type[BODY_PARTS_KPT_IDS[part_id][1]]  # 当前躯干所有终点  kpts_a和kpts_b为[]，里面可能有几个4维向量，也可能为空，几个4维向量就是几个候选起点，空则没有候选人，这个4维向量就是上面那个[x,y,score,index],debug可以看
        num_kpts_a = len(kpts_a)  # 统计这个躯干找到了几个起点
        num_kpts_b = len(kpts_b)  # 几个终点
        kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]  # 得到这个躯干的起点a在所有关节点中的索引位置，这个是上面固定的
        kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]  # 终点b在所有关节点中的索引位置，相当于得到这些终点都属于哪个关节类型

        if num_kpts_a == 0 and num_kpts_b == 0:  # 这个躯干没有检测到对应关节点
            continue
        elif num_kpts_a == 0:  # 若有关这个身体部位只检测到一头的关节b，则依次遍历终点，若终点没有被分配给任何一个人，就创建新的人
            for i in range(num_kpts_b):
                num = 0
                for j in range(len(pose_entries)):  # 看终点有没有分配给别人
                    if pose_entries[j][kpt_b_id] == kpts_b[i][3]: # 判断这个人里面的这个应存位置内的信息和这个终点一不一致
                        num += 1
                        continue
                if num == 0:  # 如果没有被分配过，就创建一个人把终点分配给他
                    pose_entry = np.ones(pose_entry_size) * -1 # 初始化
                    pose_entry[kpt_b_id] = kpts_b[i][3]  # 把此关节放进这个新的人的信息中
                    pose_entry[-1] = 1                   # 存这个人中关节点总数
                    pose_entry[-2] = kpts_b[i][2]        # 这个人的得分
                    pose_entries.append(pose_entry)  # 叠加
            continue
        elif num_kpts_b == 0:  # 这个身体部位只检测到一头的关节a，则依次遍历终点，若终点没有被分配给任何一个人，就创建新的人
            for i in range(num_kpts_a):
                num = 0
                for j in range(len(pose_entries)):
                    if pose_entries[j][kpt_a_id] == kpts_a[i][3]:
                        num += 1
                        continue
                if num == 0:
                    pose_entry = np.ones(pose_entry_size) * -1
                    pose_entry[kpt_a_id] = kpts_a[i][3]
                    pose_entry[-1] = 1
                    pose_entry[-2] = kpts_a[i][2]
                    pose_entries.append(pose_entry)
            continue
       # 以下主要就是在针对最外大循环的这个肢体类型，计算每个候选躯干的得分，最终得到的是一坨connection
        connections = []
        for i in range(num_kpts_a): # 依次遍历每个起点
            kpt_a = np.array(kpts_a[i][0:2])  # 得到这个起点的xy坐标,格式为[x,y]
            for j in range(num_kpts_b):  # 依次遍历每个终点
                kpt_b = np.array(kpts_b[j][0:2])  # 得到这个终点的xy坐标,格式为[x,y]
                mid_point = [(), ()]
                mid_point[0] = (int(round((kpt_a[0] + kpt_b[0]) * 0.5)),
                                int(round((kpt_a[1] + kpt_b[1]) * 0.5)))  # 算了一个中间点在起点和终点之间
                mid_point[1] = mid_point[0]  # mid_ponit[0]和[1]内容一样都是（x,y)即中点的坐标

                vec = [kpt_b[0] - kpt_a[0], kpt_b[1] - kpt_a[1]]  # 求由关节a到b连成的肢体的方向向量
                vec_norm = math.sqrt(vec[0] ** 2 + vec[1] ** 2)  # 向量长度
                if vec_norm == 0:
                    continue
                vec[0] /= vec_norm # 得到方向向量在x方向的单位向量
                vec[1] /= vec_norm # 得到方向向量在y方向的单位向量
                cur_point_score = (vec[0] * part_pafs[mid_point[0][1], mid_point[0][0], 0] +
                                   vec[1] * part_pafs[mid_point[1][1], mid_point[1][0], 1])  # 计算预测出来的两个关节点和paf向量的相似度

                height_n = pafs.shape[0] // 2
                success_ratio = 0
                point_num = 10  # 取10个间隔点
                if cur_point_score > -100:
                    passed_point_score = 0
                    passed_point_num = 0
                    x, y = linspace2d(kpt_a, kpt_b)  # 取起点和中点之间的均匀间隔点的坐标分别赋给x，y
                    for point_idx in range(point_num):  # 遍历每一个间隔点
                        if not demo:
                            px = int(round(x[point_idx]))  # 得到某个间隔点的x坐标的四舍五入值,为什么可以用这种方法呢
                            py = int(round(y[point_idx]))  # 得到某个间隔点的y坐标的四舍五入值
                        else:
                            px = int(x[point_idx])
                            py = int(y[point_idx])
                        paf = part_pafs[py, px, 0:2] # 得到采样的点对应的paf图上的paf向量,格式为[x,y]
                        cur_point_score = vec[0] * paf[0] + vec[1] * paf[1]  # 相当于两个向量点乘，得到两个矢量的相似度。这里vec是当前选的ab关节连起来的躯干的方向向量，paf是采样点上对应的paf向量。得到这个采样点的得分。得分相当于相似度
                        if cur_point_score > min_paf_score:  # 这个采样点的得分超过了这个阈值，就认可
                            passed_point_score += cur_point_score  # 把有效的采样点上求出来的得分加起来
                            passed_point_num += 1  # 有效点的个数加一
                    success_ratio = passed_point_num / point_num  # 插值点中大于阈值的点的数量占总插值点数量的比例
                    ratio = 0
                    if passed_point_num > 0:  # 满足要求的点的数量
                        ratio = passed_point_score / passed_point_num # 超过阈值的点的平均得分
                    ratio += min(height_n / vec_norm - 1, 0)  # 如果这个肢体太长（vec_norm太大）就不合理，那么min的结果就为负，给他再剪掉几分
                if ratio > 0 and success_ratio > 0.8:
                    score_all = ratio + kpts_a[i][2] + kpts_b[j][2]  # 这个总得分还要加上,这两个关节的坐标位置的热图得分（可理解为正确关节点的可能性）
                    connections.append([i, j, ratio, score_all])  # 每个connection就表示一个候选躯干，即边，connection的格式是[起点，终点，ratio，score_all]
        if len(connections) > 0:
            connections = sorted(connections, key=itemgetter(2), reverse=True)  # 将某个肢体类别的所有候选连接按照ratio连接置信度排序
        # 这边往下开始做匈牙利算法（针对某一个躯干类型的一坨候选躯干）比如这个最外大循环内在找左上臂，现在找到了一坨左肩和左手肘（假如各m个，实际上不一定个数一样），他们两两相连有很多个候选肢体远超m对，但最后只要留下m对候选肢体，他们之间满足二分图的概念只能一一对应，不能占用同一个点
        num_connections = min(num_kpts_a, num_kpts_b)  # 得到应有的连接数量m（这么多起点和终点实际上对应的真正的连接个数）候选肢体乱连是远超这个个数的。所以下面的操作可以视为在清理多余候选躯干
        has_kpt_a = np.zeros(num_kpts_a, dtype=np.int32)  # 初始化
        has_kpt_b = np.zeros(num_kpts_b, dtype=np.int32)  # 初始化
        filtered_connections = []  # 存储最终的匹配的连接
        for row in range(len(connections)):  # 开始清理候选躯干，最终只剩m个。因为上面sort过所以得分越高的连接先被考虑
            if len(filtered_connections) == num_connections:  # 如果已经找齐所有的连接就离开，不然就继续找
                break
            i, j, cur_point_score = connections[row][0:3] # 得到此候选躯干的起点，终点，以及躯干采样点的平均得分
            if not has_kpt_a[i] and not has_kpt_b[j]:  # 如果起点和终点都没有被占用过
                filtered_connections.append([kpts_a[i][3], kpts_b[j][3], cur_point_score])  #  将关节点a在所有关节点的序号和关节点b在所有关节点的序号，还有组成的躯干的得分添加到filtered_connections中
                has_kpt_a[i] = 1  # 这样置1就算这两个点已被使用过
                has_kpt_b[j] = 1
        connections = filtered_connections  # 因为有这一步所以这里往下的connection里面已经只剩筛选过的真正肢体了。而且格式也变成了[起点在所有关节点中的序号，终点在，躯干得分]
        if len(connections) == 0:
            continue
        # 以下在找如果有不止一个肢体左上臂，那他们应该分别是不同的人身上的，给他们对应起来。pose_entries[i]是人i，pose_entries[i][]=[20维矩阵]中[]位置的内容
        if part_id == 0:  # 如果现在是属于第一个躯干的躯干们在判断
            pose_entries = [np.ones(pose_entry_size) * -1 for _ in range(len(connections))]  # 新建m个20维的矩阵，每个表示姿态对应的人，（比如检测到m个左上臂就表示图中一定会有m个人存在）这个东西前18维为每个人各个关节点在所有关节点中的索引，最后两维分别为总分值和分配给这个人关节点的数量
            for i in range(len(connections)):  # # 依次遍历当前找到的所有左上臂躯干
                pose_entries[i][BODY_PARTS_KPT_IDS[0][0]] = connections[i][0]   # 这个属于第一个躯干类型的躯干i的起点在所有关节点中的索引值，赋给第i个人的pose entries中第BODY_PARTS_KPT_IDS[0][0]维。BODY_PARTS_KPT_IDS[0][0]是第0个躯干的起点关节索引值应该是1
                pose_entries[i][BODY_PARTS_KPT_IDS[0][1]] = connections[i][1]   # 终点在所有关节点中的索引赋给第i个人的pose entries中第（第0个躯干的终点关节索引值应该是2）维
                pose_entries[i][-1] = 2  # 当前这个人的所有关节点的数量置为2
                pose_entries[i][-2] = np.sum(all_keypoints[connections[i][0:2], 2]) + connections[i][2]  # 当前这个人的总分值=两个关节点热图得分+平均paf值
        elif part_id == 17 or part_id == 18:  # 如果是在判断最后两个躯干
            kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]  # 这个躯干的起点索引值
            kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]  # 这个躯干的终点索引值
            for i in range(len(connections)):  # 将当前躯干类型的所有躯干和part_id=0时分配的所有人依次比较。此处为当前躯干
                for j in range(len(pose_entries)):  # 此处为分配的所有人
                    if pose_entries[j][kpt_a_id] == connections[i][0] and pose_entries[j][kpt_b_id] == -1:  # 当前躯干的起点和分配到的某个人的这个躯干的起点的索引值一致，且这个人的当前躯干的终点未分配
                        pose_entries[j][kpt_b_id] = connections[i][1]  # 当前躯干的终点索引值赋给到这个人对应的pose_entries中此躯干终点应该在的位置上
                    elif pose_entries[j][kpt_b_id] == connections[i][1] and pose_entries[j][kpt_a_id] == -1:  # 当前躯干的终点和分配到的某个人的终点一致，且当前躯干的起点未分配
                        pose_entries[j][kpt_a_id] = connections[i][0]  # # 将当前躯干的起点分配到这个人对应起点上
            continue
        else:  # 如果在判断其他躯干
            kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]  # 这个躯干的起点在kpt图中的索引值
            kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]  # 终点关节在kpt图中的索引
            for i in range(len(connections)):  # 将当前躯干和所有人中对应信息依次比较。此处为当前躯干
                num = 0
                for j in range(len(pose_entries)):  # 此处为分配的所有人
                    if pose_entries[j][kpt_a_id] == connections[i][0]:  # 当前躯干的起点和分配到的某个人的起点一致
                        pose_entries[j][kpt_b_id] = connections[i][1]  # 将当前躯干的终点分配到这个人对应终点上
                        num += 1  # 分配的人+1
                        pose_entries[j][-1] += 1  # 当前人所有关节点的数量+1
                        pose_entries[j][-2] += all_keypoints[connections[i][1], 2] + connections[i][2]  # 当前人的socre增加
                if num == 0:  # 如果当前躯干没有分配到人，则再新建一个人
                    pose_entry = np.ones(pose_entry_size) * -1
                    pose_entry[kpt_a_id] = connections[i][0] # 此躯干起点索引值放在[20维中]的他应该在的位置
                    pose_entry[kpt_b_id] = connections[i][1]
                    pose_entry[-1] = 2
                    pose_entry[-2] = np.sum(all_keypoints[connections[i][0:2], 2]) + connections[i][2]
                    pose_entries.append(pose_entry)
    # 以下在在清理所有分配的人
    filtered_entries = []  # 保存最终剩下的有效人
    for i in range(len(pose_entries)):  # 依次遍历所有分配的人，
        if pose_entries[i][-1] < 4 or (pose_entries[i][-2] / pose_entries[i][-1] < 0.2):  # 如果当前人关节点数量少于3,或者当前人的平均得分小于0.2,则删除该人
            continue
        filtered_entries.append(pose_entries[i])  # 把满足条件的人加进去
    pose_entries = np.asarray(filtered_entries)  # 转换成n维数组，返回所有分配的人。最终pose_entries中存所有人和他们的关节点信息。pose_entries【】是哪个人，pose_entries【】【】是这个人的20维信息。（前18维为每个人各个关节点在所有关节点中的索引，后两为每个人得分及每个人关节点数量）
    return pose_entries, all_keypoints  # all_keypoints[]是某个关节，all_keypoints【】【】是这个关节的信息里面包含这个关节点的x，y坐标，热图得分还有在所有关节点中的索引值
