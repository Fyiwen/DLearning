from torch import nn

# 可以全删，没用到
def get_parameters(model, predicate):
    for module in model.modules():  # 迭代遍历某模型的所有子层赋给module
        for param_name, param in module.named_parameters():  #迭代的返回每个子层中的参数以及参数的名字
            if predicate(module, param_name):  # ？如果此层和给的参数名字相同
                yield param  # yield表明它返回的是一个生成器generator，这个generator可以通过迭代的方式获取所有元素


def get_parameters_conv(model, name):  # ？lambda m, p:和后面内容，定义了一个函数，这个函数对应上面的predicate（），如果此层是conv2D就返回true且给这层的group值设为1，参数名字就是传入值
    return get_parameters(model, lambda m, p: isinstance(m, nn.Conv2d) and m.groups == 1 and p == name)


def get_parameters_conv_depthwise(model, name):  # 与上同理
    return get_parameters(model, lambda m, p: isinstance(m, nn.Conv2d)
                                              and m.groups == m.in_channels
                                              and m.in_channels == m.out_channels
                                              and p == name)


def get_parameters_bn(model, name):  # 同理，匿名函数意思是这层若是bn层则返回true且把参数名设成这里传入的值
    return get_parameters(model, lambda m, p: isinstance(m, nn.BatchNorm2d) and p == name)
