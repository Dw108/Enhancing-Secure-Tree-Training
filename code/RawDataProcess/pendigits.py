from pandas import read_csv
import numpy as np
import csv
# 读取数据
df0 = read_csv("./raw/pendigits-tra.csv")
data0 = df0.values
df1 = read_csv("./raw/pendigits-tes.csv")
data1 = df1.values

# 分类任务
# 数据都是int值，不需要转化
name = ['id', 'Attribute1', 'Attribute2', 'Attribute3', 'Attribute4', 'Attribute5', 'Attribute6', 'Attribute7', 'Attribute8',
         'Attribute9', 'Attribute10', 'Attribute11', 'Attribute12', 'Attribute13', 'Attribute14', 'Attribute15', 'Attribute16', 'label', ]
tmp_a = []
for i in range(data0.shape[0]):
    tmp = data0[i]
    tmp = np.insert(tmp, 0, i)
    tmp_a.append(tmp)

for i in range(data1.shape[0]):
    tmp = data1[i]
    tmp = np.insert(tmp, 0, i + data0.shape[0])
    tmp_a.append(tmp)

with open("./raw/pendigits.csv", "w", encoding="utf-8", newline="") as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(name)
    csv_writer.writerows(tmp_a)
    f.close()