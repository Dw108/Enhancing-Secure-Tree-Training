import csv

import numpy as np
from sklearn.datasets import fetch_covtype, get_data_home
cov_type = fetch_covtype()
# print(cov_type.data[0:10])
# print(cov_type.target[0:10])
# Let's check the 4 first feature names
print(cov_type.feature_names[0:10])
print(cov_type.feature_names[10:14])
print(cov_type.feature_names[14:])
tmp_a = []
for i in range(cov_type.data.shape[0]):
    item = cov_type.data[i]
    tar = cov_type.target[i]
    tmp = item[0:10]
    Wilderness_Area = np.argmax(item[10:14])
    Soil_Type = np.argmax(item[14:])
    tmp = np.insert(tmp, 0, i)
    tmp = np.insert(tmp, tmp.size, Wilderness_Area)
    tmp = np.insert(tmp, tmp.size, Soil_Type)
    tmp = np.insert(tmp, tmp.size, tar)
    tmp_a.append(tmp)


names = cov_type.feature_names[0:10]
names.append('Wilderness_Area')
names.append('Soil_Type')
names.append('label')
names.insert(0, 'id')
with open("./dataset/covtype.csv", "a", encoding="utf-8", newline="") as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(names)
    csv_writer.writerows(tmp_a)
    f.close()