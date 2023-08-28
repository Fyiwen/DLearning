#coding=utf-8
import torch
from d2l import torch as d2l
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

torch.set_printoptions(2)  # 精简输出精度
def multibox_prior(data, sizes, ratios):
    """生成以每个像素为中心具有不同形状的锚框"""
    in_height, in_width = data.shape[-2:] # 输入图的高宽
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)  # 缩放比取值个数，宽高比取值个数
    boxes_per_pixel = (num_sizes + num_ratios - 1) # 以同一像素为中心的锚框个数
    size_tensor = torch.tensor(sizes, device=device)  # 转成张量形式,适应后面代码的需要
    ratio_tensor = torch.tensor(ratios, device=device)

    # 为了将锚点移动到像素的中心，需要设置偏移量。因为一个像素的高为1且宽为1，我们选择偏移到中心0.5
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height  # 在y轴上缩放步长
    steps_w = 1.0 / in_width  # 在x轴上缩放步长

    # 生成锚框的所有中心点，将所有中心点进行缩放处理，限制在(0,1)之间
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h  # 一列上中心点的y坐标们。正好一个像素一个 分成这些作为坐标值tensor([ 0,  1,  2,  3,  4，。。。。])再加个偏移量就是中心点，再乘上缩放倍数，就把坐标限制在01内
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w  # 一行上中心点的x坐标们
    shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij') # 得到所有中心点的y坐标和x。函数的作用是将一列上的y坐标复制到所有列，列的个数就是行长，同理复制x坐标到所有行
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)  # 将坐标们展平成一维存放，按行存储

    # 生成“boxes_per_pixel”个锚框的高和宽，之后用于创建锚框角坐标
    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
                   sizes[0] * torch.sqrt(ratio_tensor[1:])))\
                   * in_height / in_width  # 宽公式为ws*根号r，因为这么计算宽高比不会为r，所以多做了 * in_height / in_width这一步使得宽高比为r，\这个是续行符使得和下面一行逻辑连通

    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),  # s和r的组合方式不是全排列，只考虑包含s1和r1的组合，所以i锚框种类是5个
                   sizes[0] / torch.sqrt(ratio_tensor[1:])))  # 高公式为hs/根号r，                                         这里代码发现w和h根本没乘上去，怀疑因为现在的信息都是按照原比例缩放过的，所以这边要是乘上规格就不对，还得再除，约等于没有就不做了

    # 除以2得到每个候选框的左上角和右下角的坐标值的偏移量，具体来说可以将每个像素点的坐标值作为中心点，再分别加上或减去 w/2和 h/2，就可以得到对应的候选框的左上角和右下角的坐标值。（所以有两对wh，左角要-，右角要+）
    # 下面这个张量用以计算锚框的4个坐标值
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(
                                        in_height * in_width, 1) / 2 # repeat 方法将前面一行四列的矩阵重复复制成 (in_height * in_width)=像素总数 行，每行四列的矩阵。以便每个像素点都可以使用它来计算对应的锚框的坐标值。repeat函数的作用是将输入张量沿着第一个维度（即行）重复 in_height * in_width 次，沿着第二个维度（即列）重复 1 次，生成一个新的张量

    # 每个中心点都将有“boxes_per_pixel”个锚框，所有锚框信息存在一起，比如[0,n+m-1)存第一个像素的锚框[n+m-1, 2n+2m-2)存第二个像素的锚框
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y],
                dim=1).repeat_interleave(boxes_per_pixel, dim=0)#先将中心点坐标信息，沿着列拼接，变成4行，像素个数列。再沿着行重复了“boxes_per_pixel”次，现在形状为（像素总数*boxes_per_pixel，4）
    output = out_grid + anchor_manipulations # 中心+偏移后，成功得到左右角坐标值，形状为（锚框总数，4）anchor_manipulations 会自动地被沿着第一个维度（即行）重复 boxes_per_pixel 遍，以便与 out_grid 的形状相匹配
    return output.unsqueeze(0) # 增加一个维度在第0维

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
                              (boxes[:, 3] - boxes[:, 1])) # 求出框面积的函数
    # boxes1：(boxes1的数量,4),boxes2：(boxes2的数量,4),
    # areas1：(boxes1的数量),areas2：(boxes2的数量)
    areas1 = box_area(boxes1) # 所有预测框的面积
    areas2 = box_area(boxes2) # 所有真实框的面积
    # inter_upperlefts形状:(boxes1的数量,boxes2的数量,2)，结果是真实与预测框相交部分的左上角坐标
    # inter_lowerrights形状:(boxes1的数量,boxes2的数量,2)，结果是真实与预测框相交部分的右下角坐标
    # inters形状:(boxes1的数量,boxes2的数量,2)，结果是对应交集的宽高
    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2]) # None的作用就是增加一个维度，现在形状是(5,1,2),这导致前后参数维度不一致，此时就会用到广播机制根据广播机制规则，首先将boxes[:, :2]向左扩充一个维度使其形状变成(1,2,2)，现在参数都变成了三维，一个为(5,1,2),一个为(1,2,2)，所以又根据广播规则最后两个形状都会变成(5,2,2)。变成522后，里面元素格式为[[[第一个锚框x_左上，第一个锚框y_左上][第一个锚框x_左上，第一个锚框y_左上]][[第二个锚框x_左上，第二个锚框y_左上][第二个锚框x_左上，第二个锚框y_左上]]........[[第五个锚框x_左上，第五个锚框y_左上][第五个锚框x_左上，第五个锚框y_左上]]]
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0) #计算交集矩形的宽高，clamp 函数将负数部分截断，即如果两个框没有交集，则交集宽度和高度应为 0
    # inter_areasandunion_areas的形状:(boxes1的数量,boxes2的数量)
    inter_areas = inters[:, :, 0] * inters[:, :, 1] # 交集面积，形状为（boxes1的数量,boxes2的数量）
    union_areas = areas1[:, None] + areas2 - inter_areas # 并集面积。为了形状匹配，用NONE增加一维
    return inter_areas / union_areas  #得到交并比。假设锚框有m个，真实边界框（其实就代表图片中我们要识别的物体个数）有n个，则返回值(交并比)的形状(shape)为(m,n)，第i行第j列的元素含义代表，第i个锚框与第j个真实边界框的交并比

def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5): # 这个函数逻辑上和书中描述的流程相反，先给所有锚框匹配完了，再匹配那几个最接近真实框的
    """给锚框们分配真实边缘框，来得到锚框的预测类别啊，偏移量等"""
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0] # 预测框和真实框的个数
    jaccard = box_iou(anchors, ground_truth) # 计算所有真实框和预测框对应的IOU，结果中位于第i行和第j列的元素x_ij是锚框i和真实边界框j的IoU

    # 对于每个锚框，分配的真实边界框的张量，设索引为idx，代表第idx+1个锚框，索引idx对应的value(0或1)为value+1个真实边界框
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long,
                                  device=device)  # 初始化大小为(num_anchors)的张量，初始值为-1，用于记录每个锚框匹配的最佳真实框的索引

    # 根据阈值，决定是否分配真实边界框
    max_ious, indices = torch.max(jaccard, dim=1) #每一行的最大值和其对应的列索引
    anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1) # 返回max_ious中大于0.5的元素索引的下标，即锚框索引。torch.nonzero返回其中非零元素的索引
    box_j = indices[max_ious >= iou_threshold] # 返回max_ious中大于0.5的元素的列索引下标，即真实框索引
    anchors_bbox_map[anc_i] = box_j # 记录匹配的框们，锚框索引号对应的位置存其匹配的真实框索引

    col_discard = torch.full((num_anchors,), -1) # 初始化
    row_discard = torch.full((num_gt_boxes,), -1)
    for _ in range(num_gt_boxes): # 遍历每一个真实边界框
        max_idx = torch.argmax(jaccard) # 找到IOU矩阵中最大值的索引，是全局的索引，所以可以通过以下计算得到对应行列索引
        box_idx = (max_idx % num_gt_boxes).long() # 取列索引
        anc_idx = (max_idx / num_gt_boxes).long() # 取行索引
        anchors_bbox_map[anc_idx] = box_idx #保存新匹配的内容
        # 把匹配到的相关行列置-1
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    return anchors_bbox_map # 返回所有匹配信息

def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """锚框到分配的真实边框的偏移量计算"""
    c_anc = d2l.box_corner_to_center(anchors) # 转变锚框信息的表示方式，变成中心、宽高那种，形状为（n,4)
    c_assigned_bb = d2l.box_corner_to_center(assigned_bb) # 分配的真实边界框转换格式
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:] # 计算中心点偏移量，*10是为了放大偏移量的影响，以便更好地训练模型.[:, :2]所有锚框xy信息
    offset_wh = 5 * torch.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:]) # 计算真实边界框的宽高与锚框的宽高的比例，并取对数。得到偏移量的宽高，eps是为了避免取对数时得到无穷大。[:, 2:]宽高信息
    offset = torch.cat([offset_xy, offset_wh], dim=1) # 拼接所有偏移量信息
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





def offset_inverse(anchors, offset_preds): # offset_preds是预测出来的相对于先验框的偏移量（类别数，4），anchor（先验框数量，4）先验框是指一些预先定义好的边界框，用于在图像中覆盖可能包含待检测物体的区域，相当于真实框
    """根据预测偏移量和真实框的信息还原出预测边界框"""
    anc = d2l.box_corner_to_center(anchors)

    # 以下式子对应上面的offset_boxes函数计算，相当于在还原信息
    pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2] # 表示预测的边界框在图像中的左上角坐标，(N, 2)
    pred_bbox_wh = torch.exp(offset_preds[:, 2:] / 5) * anc[:, 2:] # 预测的边界框的宽高信息
    pred_bbox = torch.cat((pred_bbox_xy, pred_bbox_wh), dim=1) # 连接成一个整的预测框信息
    predicted_bbox = d2l.box_center_to_corner(pred_bbox) # 转成角表示法
    return predicted_bbox

def nms(boxes, scores, iou_threshold):
    """对预测边界框的置信度进行排序"""
    B = torch.argsort(scores, dim=-1, descending=True) # 在最后一维降序排列，返回的是这些排序好的置信度的原来索引值
    keep = []  # 记录被保留的预测边界框的索引

    # 下面就是实现NMS这个逻辑，先找置信度第一大记下后排掉和他重合高的，再找第二大，知道所有框都被安排到
    while B.numel() > 0: # B中元素个数只要有
        i = B[0] # 最前面位置的内容赋给i，i是当前拥有最高置信度的边界框的索引
        keep.append(i) # 这个索引的边界框被记录下来保留
        if B.numel() == 1: break # 这是最后一个，框都判断完了，就可以结束了
        iou = box_iou(boxes[i, :].reshape(-1, 4), # 形状为1*4 ，-1就是自己计算出剩下的另一个shape属性
                      boxes[B[1:], :].reshape(-1, 4)).reshape(-1) # 计算当前框和其他没保留的锚框的IOU，形状为（n-1）*4
        #inds = torch.nonzero(iou <= iou_threshold).reshape(-1)# 和当前这个重叠不大的框被留下，继续判断
        inds = torch.nonzero(iou <= 0.5).reshape(-1)  # 和当前这个重叠不大的框被留下，继续判断
        B = B[inds + 1] #加一是因为算iou时记录的框索引时从第二个预测边界框开始的但是记为0，所以还原到这里，索引值加一才是本来这里的索引
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













