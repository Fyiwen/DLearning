#coding=utf-8
import torch
from d2l import torch as d2l
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

torch.set_printoptions(2)  # 精简输出精度
def multibox_prior(data, sizes, ratios):

    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = torch.tensor(sizes, device=device)
    ratio_tensor = torch.tensor(ratios, device=device)

    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height
    steps_w = 1.0 / in_width

    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
    shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij')
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
                   sizes[0] * torch.sqrt(ratio_tensor[1:])))\
                   * in_height / in_width

    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),
                   sizes[0] / torch.sqrt(ratio_tensor[1:])))

    # 除以2得到每个候选框的左上角和右下角的坐标值的偏移量，具体来说可以将每个像素点的坐标值作为中心点，再分别加上或减去 w/2和 h/2，就可以得到对应的候选框的左上角和右下角的坐标值。（所以有两对wh，左角要-，右角要+）
    # 下面这个张量用以计算锚框的4个坐标值
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(
                                        in_height * in_width, 1) / 2
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y],
                dim=1).repeat_interleave(boxes_per_pixel, dim=0)
    output = out_grid + anchor_manipulations
    return output.unsqueeze(0)

img = d2l.plt.imread('D:/pycode/xianxinhuigui/data/img/catdog.jpg')
h, w = img.shape[:2]
print(h, w)
X = torch.rand(size=(1, 3, h, w))
Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
print(Y.shape)  # (批量大小，锚框的数量，4) 锚框总数=h*w*5，有那么多像素，每个像素还要对应5个，4里面是角坐标信息

boxes = Y.reshape(h, w, 5, 4)  # 将锚框变量Y的形状更改为(图像高度,图像宽度,以同一像素为中心的锚框的数量,4)，相当于把所有锚框信息整理了一下，方便下面读取
print(boxes[250, 250, 0, :])  # 访问以（250,250）为中心的第一个锚框。 它有四个元素：锚框左上角的轴坐标和右下角的轴坐标。 输出中两个轴的坐标各分别除以了图像的宽度和高度。

def show_bboxes(axes, bboxes, labels=None, colors=None):
    """显示以图像中以某个像素为中心的所有锚框"""
    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)): # 如果 obj 不是列表或元组，就将它转换成一个只含有一个元素的列表。这个函数是为了方便处理输入参数。
            obj = [obj]
        return obj

    labels = _make_list(labels) # 转换成列表
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c']) # 如果 colors 没有被传入，就使用默认值 ['b', 'g', 'r', 'm', 'c']
    for i, bbox in enumerate(bboxes): # 遍历每个边缘框
        color = colors[i % len(colors)] # 选择一种颜色，尽量保证一个像素对应的锚框颜色不重复
        rect = d2l.bbox_to_rect(bbox.detach().numpy(), color) # 绘制矩形边框
        axes.add_patch(rect) # 将这个矩形对象添加到当前 Axes 对象中。
        if labels and len(labels) > i: # 如果标签列表 labels 不为空，labels 中的标签数量大于当前循环变量 i
            text_color = 'k' if color == 'w' else 'w' # 选择文字颜色
            axes.text(rect.xy[0], rect.xy[1], labels[i], # 文本左下角xy坐标
                      va='center', ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0)) # 在矩形中央添加一个标签。 # 文本背景框，填充色，边框宽度0
d2l.set_figsize()
bbox_scale = torch.tensor((w, h, w, h))  # 用于恢复原始坐标值
fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, boxes[250, 250, :, :] * bbox_scale,  # 绘制出图像中所有以(250,250)为中心的锚框，这部分代码乘bbox_scale就是前面对求锚框时对其进行了缩放处理，现在要放大为原来的样子
            ['s=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1', 's=0.75, r=2',
             's=0.75, r=0.5'])




def box_iou(boxes1, boxes2):  #boxes1：一张图片中的所有锚框，boxes2：一张图片中的所有真实边界框
    """计算两个锚框或边界框列表中成对的交并比"""
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)

    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)

    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas

def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """给锚框们分配真实边缘框，来得到锚框的预测类别啊，偏移量等"""
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    jaccard = box_iou(anchors, ground_truth)

    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long,
                                  device=device)

    max_ious, indices = torch.max(jaccard, dim=1)
    anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1)
    box_j = indices[max_ious >= iou_threshold]
    anchors_bbox_map[anc_i] = box_j

    col_discard = torch.full((num_anchors,), -1)
    row_discard = torch.full((num_gt_boxes,), -1)
    for _ in range(num_gt_boxes):
        max_idx = torch.argmax(jaccard)
        box_idx = (max_idx % num_gt_boxes).long()
        anc_idx = (max_idx / num_gt_boxes).long()
        anchors_bbox_map[anc_idx] = box_idx

        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    return anchors_bbox_map

def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """锚框到分配的真实边框的偏移量计算"""
    c_anc = d2l.box_corner_to_center(anchors)
    c_assigned_bb = d2l.box_corner_to_center(assigned_bb)
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wh = 5 * torch.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    offset = torch.cat([offset_xy, offset_wh], dim=1)
    return offset

def multibox_target(anchors, labels): # anchors：其shape为(1,anchor_num,4)，anchor_num表示一张图片中所有锚框数量，4里面是(左上，右下)方法表示； labels：其shape为(batch_size, class_num, 5)，5是类别信息+上下标
    """使用真实边界框标记锚框"""
    batch_size, anchors = labels.shape[0], anchors.squeeze(0) # anchors形状变为（anchor_num,4）
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]
    for i in range(batch_size): # 遍历每一个批次中的所有样本
        label = labels[i, :, :] # 提取当前批次中的所有标签信息，形状为（class_num, 5）class_num是要预测的类别数，也就是说有这个多个真实边缘框
        anchors_bbox_map = assign_anchor_to_bbox(
            label[:, 1:], anchors, device) # label[:, 1:]是所有真实边缘框的框坐标信息，最终得到所有锚框匹配信息

        #下面这个参数形状为(num_anchors, 4)，其中每一行的4个元素表示对应锚框的4个偏移量是否需要参与计算损失函数。1用0不用。用处在下面
        # 如果对应位置的anchors_bbox_map值为-1，表示未匹配到真框则这四个偏移量将被忽略。如果对应位置的anchors_bbox_map值不为-1，则这四个偏移量将参与计算损失函数。
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(1, 4) # 判断每个锚框有没有匹配到的真实边界框的索引，没有匹配到的内容值是-1<0，判断后经过float().unsqueeze(-1)变成形状为(num_anchors, 1)的浮点数张量，unsqueeze(-1)在最后一维加1，其中匹配到真实边界框的位置为1，否则为0,将这个张量在最后一个维度上重复4次，以便与偏移量张量的形状相同

        # 将类标签和分配的边界框坐标初始化为零
        class_labels = torch.zeros(num_anchors, dtype=torch.long,
                                   device=device)
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32,
                                  device=device)

        # 如果一个锚框没有被分配，标记其为背景（值为零）
        indices_true = torch.nonzero(anchors_bbox_map >= 0) # 所有满足阈值的非零位置的索引，即所有被分配到真实框的锚框的索引
        bb_idx = anchors_bbox_map[indices_true] # 所有被分配到真实框的锚框的对应真实框的索引
        class_labels[indices_true] = label[bb_idx, 0].long() + 1 # 记下锚框的匹配的真实框的类标签，是哪个类，+1是因为索引从0开始，类别从1
        assigned_bb[indices_true] = label[bb_idx, 1:] # 记下锚框的匹配的真实框的4个角坐标

        offset = offset_boxes(anchors, assigned_bb) * bbox_mask #计算锚框和真框的偏移量，这里用了mask只计算那些匹配到的框偏移量
        batch_offset.append(offset.reshape(-1)) # 一张图中所有偏移量，形状num_anchors*4
        batch_mask.append(bbox_mask.reshape(-1)) # 掩码，形状num_anchors*4，表示其偏移量是不是有效的，要不要考虑的
        batch_class_labels.append(class_labels) #一张图中的所有锚框的类别标签，形状num_anchors
    bbox_offset = torch.stack(batch_offset) # shape为(batch_size, ele_num)第1维度存一张图片中的偏移量，是4个元素为一组，每组为一个锚框对应的偏移量。ele_num = anchor_num*4
    bbox_mask = torch.stack(batch_mask) # shape为(batch_size, ele_num)，第1维度每四个元素为一组表示该锚框是否为背景。若为背景，则四个元素均为0，否则均为1
    class_labels = torch.stack(batch_class_labels) # shape为(anchor_num,)，值为0表示背景，否则表示某类别的物体(例如等于1代表一种物体，等于2又代表一种物体)
    return (bbox_offset, bbox_mask, class_labels)

# 实例，猫狗图
ground_truth = torch.tensor([[0, 0.1, 0.08, 0.52, 0.92],
                         [1, 0.55, 0.2, 0.9, 0.88]]) #定义真实边界框，其中第一个元素是类别（0代表狗，1代表猫），其余四个元素是左上角和右下角的轴坐标（范围介于0和1之间）
anchors = torch.tensor([[0, 0.1, 0.2, 0.3], [0.15, 0.2, 0.4, 0.4],
                    [0.63, 0.05, 0.88, 0.98], [0.66, 0.45, 0.8, 0.8],
                    [0.57, 0.3, 0.92, 0.9]])  # 构建了五个锚框，用左上角和右下角的坐标进行标记
# 在图像中绘制这些真实边界框和锚框
fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, ground_truth[:, 1:] * bbox_scale, ['dog', 'cat'], 'k')
show_bboxes(fig.axes, anchors * bbox_scale, ['0', '1', '2', '3', '4'])

# 根据狗和猫的真实标签边界框，标注这些锚框的类别和偏移量。 背景、狗和猫的类索引分别为0、1和2
labels = multibox_target(anchors.unsqueeze(dim=0),            # 在第0维增加一维，配合函数输入形状
                         ground_truth.unsqueeze(dim=0))
# 在所有的锚框和真实边界框配对中，锚框4与猫的真实边界框的IoU是最大的。 因此，4的类别被标记为猫。 去除包含4或猫的真实边界框的配对，在剩下的配对中，锚框1和狗的真实边界框有最大的IoU。 因此，1的类别被标记为狗。 接下来，我们需要遍历剩下的三个未标记的锚框：0，2，3对于0与其拥有最大IoU的真实边界框的类别是狗，但IoU低于预定义的阈值（0.5），因此该类别被标记为背景； 对于2与其拥有最大IoU的真实边界框的类别是猫，IoU超过阈值，所以类别被标记为猫； 对于3与其拥有最大IoU的真实边界框的类别是猫，但值低于阈值，因此该类别被标记为背景
print(labels[0])  # 输出为每个锚框标记的四个偏移值。 负类锚框的偏移量被标记为零
print(labels[1]) # 输出掩码（mask）变量，形状为（批量大小，锚框数的四倍）。 掩码变量中的元素与每个锚框的4个偏移量一一对应。 由于我们不关心对背景的检测，负类的偏移量不应影响目标函数。 通过元素乘法，掩码变量中的零将在计算目标函数之前过滤掉负类偏移量
print(labels[2]) # 输出5个锚框对应的类别标签





def offset_inverse(anchors, offset_preds):
    """根据预测偏移量和真实框的信息还原出预测边界框"""
    anc = d2l.box_corner_to_center(anchors)
    pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
    pred_bbox_wh = torch.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
    pred_bbox = torch.cat((pred_bbox_xy, pred_bbox_wh), dim=1)
    predicted_bbox = d2l.box_center_to_corner(pred_bbox)
    return predicted_bbox

def nms(boxes, scores, iou_threshold):
    """对预测边界框的置信度进行排序"""
    B = torch.argsort(scores, dim=-1, descending=True)
    keep = []

    while B.numel() > 0:
        i = B[0]
        keep.append(i)
        if B.numel() == 1: break
        iou = box_iou(boxes[i, :].reshape(-1, 4),
                      boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
        inds = torch.nonzero(iou <= 0.5).reshape(-1)
        B = B[inds + 1]
    return torch.tensor(keep, device=boxes.device)

def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5,
                       pos_threshold=0.009999999): # 预测概率，shape（batch_size,类别数，锚框数），预测偏移量，锚框信息，。。
    """使用非极大值抑制来预测边界框"""
    device, batch_size = cls_probs.device, cls_probs.shape[0]
    anchors = anchors.squeeze(0) # 减少掉第0维
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size): # 遍历批次中每一张图，这里只有一张
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4) # cls_prob这个图所有锚框的预测概率信息shape为（锚框数，类别数），offset_pred这个图所有锚框预测偏移量（类别数，锚框数，4）
        conf, class_id = torch.max(cls_prob[1:], 0) # 得到预测概率最高的非背景类别和其类别索引。。max（x，0）找最大，干掉第0个维度。按照列找最大值返回行索引
        predicted_bb = offset_inverse(anchors, offset_pred) # 根据预测的偏移量和锚框的位置计算出预测框的坐标
        keep = nms(predicted_bb, conf, nms_threshold) # 使用非极大值抑制算法（NMS）筛选出最终的预测框

        # 找到所有的未保留的框
        all_idx = torch.arange(num_anchors, dtype=torch.long, device=device) # 生成一个长度为num_anchors的张量all_idx，表示所有的锚框索引
        combined = torch.cat((keep, all_idx)) # 将NMS保留的框的索引和all_idx拼接在一起，，，。，并按照索引排序
        uniques, counts = combined.unique(return_counts=True) # 去除重复元素，并且得到一个新列表。return_counts为 true，会返回去重数组中的元素在原数组中的出现次数
        non_keep = uniques[counts == 1] # 通过unique函数获取到仅出现一次的索引即未被NMS保留的框的索引，记录下来
        all_id_sorted = torch.cat((keep, non_keep)) # 将保留的框和未被保留的框的索引拼接在一起
        class_id[non_keep] = -1 # 把未保留的锚框的类别设为-1背景
        class_id = class_id[all_id_sorted] # 重新排布信息，现在存储的前一部分是保留框的类别，后一部分是未保留框的类别
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted] # 和上面class_id一样内部信息对应重排列
        # pos_threshold是一个用于非背景预测的阈值
        below_min_idx = (conf < pos_threshold) # 置信度不满足阈值的锚框索引
        class_id[below_min_idx] = -1 # 对应锚框也视为背景类别
        conf[below_min_idx] = 1 - conf[below_min_idx] # 因为现在其属于背景类，不像以前预测他属于别的类，所以置信度变大了，值和之前的互补
        pred_info = torch.cat((class_id.unsqueeze(1),# （n,1)
                               conf.unsqueeze(1), # (n,1)
                               predicted_bb), dim=1) #(n,4)现在 形状为(n,6)
        out.append(pred_info)
    return torch.stack(out)# 形状为（批量大小，锚框的数量，6）

#应用到一个带有四个锚框的具体示例中。 为简单假设预测的偏移量都是零，这意味着预测的边界框即是锚框。
anchors = torch.tensor([[0.1, 0.08, 0.52, 0.92], [0.08, 0.2, 0.56, 0.95],
                      [0.15, 0.3, 0.62, 0.91], [0.55, 0.2, 0.9, 0.88]])
offset_preds = torch.tensor([0] * anchors.numel()) # 自定义一个预测的偏移量，全0
# 对于背景、狗和猫其中的每个类，定义了每个锚框对他们的预测概率。3大类，4个锚框
cls_probs = torch.tensor([[0] * 4,  # 背景的预测概率
                      [0.9, 0.8, 0.7, 0.1],  # 狗的预测概率
                      [0.1, 0.2, 0.3, 0.9]])  # 猫的预测概率
fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, anchors * bbox_scale,
            ['dog=0.9', 'dog=0.8', 'dog=0.7', 'cat=0.9']) # 图像中绘制
output = multibox_detection(cls_probs.unsqueeze(dim=0),
                            offset_preds.unsqueeze(dim=0),
                            anchors.unsqueeze(dim=0),
                            nms_threshold=0.5)
print(output)  # （批量大小，锚框的数量，6）最内层维度中的6个元素提供了每一个预测边界框的输出信息。 第一个元素是预测的类索引，从0开始（0代表狗，1代表猫），值-1表示背景或在非极大值抑制中被移除了。 第二个元素是预测的边界框的置信度。 其余四个元素分别是预测边界框左上角和右下角坐标轴坐标（范围介于0和1之间）

# 删除-1类别（背景）的预测边界框后，我们可以输出由非极大值抑制保存的最终预测边界框
fig = d2l.plt.imshow(img)
for i in output[0].detach().numpy():# 这里每个i对应一行6列的内容，即某一个锚框的6个信息
    if i[0] == -1: # 这个锚框的类别是背景，掠过
        continue
    label = ('dog=', 'cat=')[int(i[0])] + str(i[1]) # 如果int(i[0])值是0则取元组中的第一个字符串'dog='，否则取元组中的第二个字符串'cat='，后面再连接字符str(i[1])即置信度
    show_bboxes(fig.axes, [torch.tensor(i[2:]) * bbox_scale], label) # i[2:]框信息













