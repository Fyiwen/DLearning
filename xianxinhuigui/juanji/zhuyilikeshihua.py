import torch
from d2l import torch as d2l
'''可视化注意力权重'''
def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5),
                  cmap='Reds'): # matrices的形状是 （要显示的行数，要显示的列数，查询的数目，键的数目）,cmap表示颜色映射
    """显示矩阵热图"""
    d2l.use_svg_display() # 将绘制图片保存成SVG格式
    num_rows, num_cols = matrices.shape[0], matrices.shape[1] # 输入矩阵的行数和列数
    fig, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize,
                                 sharex=True, sharey=True, squeeze=False) # 创建一个大小为(num_rows, num_cols)的子图，figsize参数表示热图的大小，sharex和sharey参数均设为True表示共享x轴和y轴，squeeze参数设为False则表示不会自动扁平化形状
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)): # enumerate函数对每一行绘制的子图对象进行迭代，并使用zip函数将输入矩阵与子图列表一一对应
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)): # enumerate函数对每列绘制的子图对象进行迭代，并使用zip函数将热图对象与子图对象一一对应
            pcm = ax.imshow(matrix.detach().numpy(), cmap=cmap) # 使用imshow函数在当前绘制的子图对象上，绘制输入矩阵并赋值给pcm变量
            if i == num_rows - 1: # 该行判断是否为最后一行，若是则为当前子图对象设置X轴标签
                ax.set_xlabel(xlabel)
            if j == 0: # 判断是否为第一列，若是则为当前子图对象设置Y轴标签
                ax.set_ylabel(ylabel)
            if titles: # 判断是否不存在子图标题，若是则为当前子图对象设置标题
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6) # 为绘制的子图添加颜色条

# 在本例子中，仅当查询和键相同时，注意力权重为1，否则为0
attention_weights = torch.eye(10).reshape((1, 1, 10, 10)) # torch.eye(10)生成10*10的单位矩阵
show_heatmaps(attention_weights, xlabel='Keys', ylabel='Queries')