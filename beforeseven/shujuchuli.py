import os
import numpy as np
import pandas as pd
import torch
from numpy import nan as NaN

os.makedirs(os.path.join('..', 'data'), exist_ok=True)  # 创建文件夹
datafile = os.path.join('..', 'data', 'house_tiny.csv')  # 创建文件
with open(datafile,'w') as f:  # 往文件中写数据
    f.write('rooms, alley, price\n')
    f.write('NA,pave,129300\n')
    f.write('2,NA,103200\n')
    f.write('5,NA,1235444\n')
    f.write('NA,NA,143200\n')

data = pd.read_csv(datafile)  # 读取文件内容
print("1.原始数据：\n", data)  # 原始表格中的NA被识别为NAN

inputs, outputs = data.iloc[:, 0: 2], data.iloc[:, 2]  # 第0列和第一列内容赋给in，第二列内容赋给out
print(inputs)
inputs = inputs.fillna(inputs.mean(numeric_only=True))  # 为NAN的地方填值，值为inputs内容中值元素的平均值
print(inputs)
print(outputs)

inputs = pd.get_dummies(inputs, dummy_na=True)  # 处理离散值或类别值，这里将NAN视为一个类别，那么pave就是另一个类别
print("2.利用函数处理：\n", inputs)

x, y = torch.tensor(inputs.values), torch.tensor(outputs.values)  # in，out中的元素值赋给xy
print("3.转换为张量：")
print(x, y)

# 扩展填充函数fillna的用法
df1 = pd.DataFrame([[1, 2, 3], [NaN, NaN, 2], [NaN, NaN, NaN], [8, 8, NaN]])  # 创建初始数据
print('4.函数fillna的用法：')
print(df1)
print(df1.fillna(100))  # 用常数填充 ，默认不会修改原对象
print(df1.fillna({0: 10, 1: 20, 2: 30}))  # 通过字典填充不同的常数，默认不会修改原对象
print(df1.fillna(method='ffill'))  # 用前面的值来填充
# print(df1.fillna(0, inplace=True))  # inplace= True直接修改原对象，否则会创建一个副本内容是修改后

df2 = pd.DataFrame(np.random.randint(0, 10, (5, 5)))  # 随机创建一个5*5表格
df2.iloc[1:4, 3] = NaN  # 第1行到第三行的，第三列上填上
df2.iloc[2:4, 4] = NaN  # 指定的索引处插入值
print(df2)
print(df2.fillna(method='bfill', limit=2))  # 用下一个（同列下行）填充，即下面那个，限制填充个数每列只填充两个，因为没指明axis所以默认按列
print(df2.fillna(method="ffill", limit=1, axis=1))  # 用上一个（同列上行），只填充一个，按行来填充，即每行填充一个