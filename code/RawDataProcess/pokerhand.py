from pandas import read_csv
import numpy as np
import csv
# 读取数据
df0 = read_csv("./raw/poker-hand-training-true.data")
data0 = df0.values

# 分类任务
# 数据都是int值，不需要转化
name = ['id', 'S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5', 'label', ]
tmp_a = []
for i in range(data0.shape[0]):
    tmp = data0[i]
    tmp = np.insert(tmp, 0, i)
    tmp_a.append(tmp)



with open("./raw/pokerhand.csv", "w", encoding="utf-8", newline="") as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(name)
    csv_writer.writerows(tmp_a)
    f.close()