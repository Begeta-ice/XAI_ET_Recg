import pandas as pd
import numpy
import numpy as np
import xlwt  # 负责写excel

# columns_names = ['1', '2', '3', '4']
# data = pd.read_csv("E:/etdata.csv", names=columns_names)

my_matrix = numpy.loadtxt(open("E:\\newdata.csv", "rb"), delimiter=",", skiprows=0)


def seg(dataset):
    dataset_list = np.empty((0, 512))
    ds = dataset.shape[0]  # 行数

    i = 0  # 循环初始条件
    while i < ds:
        if i + 128 <= ds:
            dl = dataset[i:i + 128, :]
            x1 = dl[:, 0:1]
            y1 = dl[:, 1:2]
            z1 = dl[:, 2:3]
            a = dl[:, 3:4]
            dataset_list = np.vstack((dataset_list, np.hstack([x1.T, y1.T, z1.T, a.T])))

        i = i + 32


    return dataset_list

result = seg(my_matrix)
print(result)

Data = pd.DataFrame(result)
# 将结果写入Excel
writer = pd.ExcelWriter("E:\\newsegment.xlsx")  # 写入Excel文件
Data.to_excel(writer, 'Sheet1')  # ‘Sheet1’是写入excel的sheet名
writer.save()

