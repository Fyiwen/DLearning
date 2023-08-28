from torch import nn

# 定义要使用的3种基础型卷积神经网络
def conv(in_channels, out_channels, kernel_size=3, padding=1, bn=True, dilation=1, stride=1, relu=True, bias=True):  # 默认使用bn归一化层和relu激活层，dilation=1等同于没有dilation的标准卷积，一旦变成例如2，感受野会发生膨胀（类似stride但是sride带着整个感受野跳跃，dilation是把一个紧凑的感受野的小方块都拉开，总面积不大但是边缘变大
    modules = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)] # 根据参数值定义了一个网络模型
    if bn:  # 如果传入参数选择了bn层
        modules.append(nn.BatchNorm2d(out_channels))  # 就在刚刚模型的基础上增加bn层
    if relu:   # 如果传入参数选择了relu层
        modules.append(nn.ReLU(inplace=True))  # 就在刚刚模型的基础上增加relu层，并且这里选择会改变输入数据的值
    return nn.Sequential(*modules)  # 返回完整的网络模型


def conv_dw(in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1):  # 又一个网络模型，这是MobileNet网络结构，因为我这个openpose算是在原来openpose基础上改过的轻量级，所以把VGG也改掉了
    return nn.Sequential(  # 设了group参数，将输入通道和输出通道进行分组，这里group=inchannel直接把每个输入通道分为一组，且这里输入输出通道数相同，比如为24，group大小也为24，那么每个输出卷积核，只与输入的对应的通道进行卷积。理解：group把input_channel(输入通道)分成group组，每组生成output_channel/group个feature_map，若不是前面那种分在一起的极端情况例子1：input_channel=6,output_channel=18,当group=2时，把input_channel分成2组，每组3个通道；同时把output_channel分成2组，每组9个，那么第一组的feature_map是通过第一组的3个input_channel生成9个feature_map，相当于每次有一个卷积核对第一组的三个通道对应信息作卷积得到三个结果，合并成一个，成为一张特征图。
        nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation=dilation, groups=in_channels, bias=False),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),

        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def conv_dw_no_bn(in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation=dilation, groups=in_channels, bias=False),
        nn.ELU(inplace=True),

        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        nn.ELU(inplace=True),  # elu是对relu函数的改进型
    )
