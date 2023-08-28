from math import pi

#实现人体关键点防抖动滤波器——一欧元滤波器

def get_alpha(rate=30, cutoff=1):  # 为得到平滑因子的函数，为下面准备。rate是采样频率默认30
    tau = 1 / (2 * pi * cutoff)  # cutoff是截止频率，得到τ是使用截止频率计算的时间常数
    te = 1 / rate  # 得到采样周期
    return 1 / (1 + tau / te)  # 得到平滑因子


class LowPassFilter:  # 利用指数平滑得到滤波后的信号的函数，为下面准备
    def __init__(self):
        self.x_previous = None

    def __call__(self, x, alpha=0.5):  # 平滑因子默认为0.5
        if self.x_previous is None:
            self.x_previous = x
            return x
        x_filtered = alpha * x + (1 - alpha) * self.x_previous # 滤波后的信号
        self.x_previous = x_filtered
        return x_filtered


class OneEuroFilter:  # 正式做一欧元滤波器实现抖动和延迟的平衡
    def __init__(self, freq=15, mincutoff=1, beta=0.05, dcutoff=1):  # 初始化参数
        self.freq = freq  # 采样频率，1/freq=Te
        self.mincutoff = mincutoff  # 默认最小截止频率为1(经验所得）
        self.beta = beta  # 速度系数，还看到说经验上0.007更好，这个可以后面试一试
        self.dcutoff = dcutoff  # 默认恒定截止频率为1
        self.filter_x = LowPassFilter()
        self.filter_dx = LowPassFilter()
        self.x_previous = None
        self.dx = None

    def __call__(self, x):
        if self.dx is None:
            self.dx = 0
        else:
            self.dx = (x - self.x_previous) * self.freq  # 过滤后的变化率
        dx_smoothed = self.filter_dx(self.dx, get_alpha(self.freq, self.dcutoff))  # 用恒定截止频率算出平滑之后的速度
        cutoff = self.mincutoff + self.beta * abs(dx_smoothed)  # 求出截止频率，根据最小截止频率，速度系数和平滑后速度
        x_filtered = self.filter_x(x, get_alpha(self.freq, cutoff))  # 用截止频率算出速度
        self.x_previous = x
        return x_filtered


if __name__ == '__main__':
    filter = OneEuroFilter(freq=15, beta=0.1)
    for val in range(10):
        x = val + (-1)**(val % 2)
        x_filtered = filter(x)
        print(x_filtered, x)
