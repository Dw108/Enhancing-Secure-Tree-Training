from pandas import read_csv
import numpy as np
import csv
# 读取数据
df0 = read_csv("./raw/Folds5x2_pp.csv")
data0 = df0.values


# 回归任务
# 浮点数数据，需要转为int
name = ['id', 'AT', 'V', 'AP', 'RH', 'target']
tmp_a = []
for i in range(data0.shape[0]):
    tmp = data0[i]
    tmp = np.insert(tmp, 0, i)
    tmp_a.append(tmp)



with open("./raw/CombinedCyclePowerPlant.csv", "w", encoding="utf-8", newline="") as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(name)
    csv_writer.writerows(tmp_a)
    f.close()