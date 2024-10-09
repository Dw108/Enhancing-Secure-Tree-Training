from pandas import read_csv
import numpy as np
import csv
# 读取数据
df0 = read_csv("./raw/CASP.csv")
data0 = df0.values

name_tmp = np.array(df0.columns.array)
name = name_tmp[1:name_tmp.size]
name = np.insert(name, name.size, 'target')
name = np.insert(name, 0, 'id')

# 回归任务
# 数据为浮点数，需要转化为int
tmp_a = []
for i in range(data0.shape[0]):
    tmp = data0[i]
    data_tmp = tmp[1:]
    label = tmp[0]
    tmp = np.insert(data_tmp, data_tmp.size, label)
    tmp = np.insert(tmp, 0, i)
    tmp_a.append(tmp)



with open("./raw/CASP_processed.csv", "w", encoding="utf-8", newline="") as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(name)
    csv_writer.writerows(tmp_a)
    f.close()