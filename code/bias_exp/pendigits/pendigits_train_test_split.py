# 训练集、测试集7：2：1
import copy
import numpy as np
from scipy.stats import entropy, gaussian_kde
from numpy import transpose



import csv
csv_reader = csv.reader(open("../../../dataset/pendigits/pendigits.csv"))
tmp = []
for row in csv_reader:
    tmp.append(row)
name = tmp[0]

arr = tmp[1:]
arr = arr[0:]
arr = np.array(arr)
np.random.shuffle(arr)
arr = arr[0:]


split_1 = int(len(arr)*0.7)
split_2 = int(len(arr)*0.2)
arr_train = arr[0:split_1]
arr_validate = arr[split_1:split_1+split_2]
arr_test = arr[split_1+split_2:]

# 无重叠数据
with open("../../../dataset/test/pen_train.csv", "w", encoding="utf-8", newline="") as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(name)
    csv_writer.writerows(arr_train)
    f.close()
with open("../../../dataset/test/pen_validate.csv", "w", encoding="utf-8", newline="") as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(name)
    csv_writer.writerows(arr_validate)
    f.close()
with open("../../../dataset/test/pen_test.csv", "w", encoding="utf-8", newline="") as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(name)
    csv_writer.writerows(arr_test)
    f.close()

# 创建子属性的方式
name_feature = copy.deepcopy(name)
# name_feature.insert(2, 'Elevation_sub')
# name_feature.insert(4, 'Aspect_sub')
name_feature.insert(2, 'F1_sub')
name_feature.insert(4, 'F2_sub')
train_sub_feature = []
validate_sub_feature = []
test_sub_feature = []
for i in range(len(arr_train)):
    tmp = arr_train[i].astype(float)
    tmp = np.insert(tmp, 2, float(tmp[1]) + np.random.laplace(loc=0, scale=10/1))
    tmp = np.insert(tmp, 4, float(tmp[3]) + np.random.laplace(loc=0, scale=10/1))
    train_sub_feature.append(tmp)

for i in range(len(arr_validate)):
    tmp = arr_validate[i].astype(float)
    tmp = np.insert(tmp, 2, float(tmp[1]) + np.random.laplace(loc=0, scale=10/1))
    tmp = np.insert(tmp, 4, float(tmp[3]) + np.random.laplace(loc=0, scale=10/1))
    validate_sub_feature.append(tmp)

for i in range(len(arr_test)):
    tmp = arr_test[i].astype(float)
    tmp = np.insert(tmp, 2, float(tmp[1]) + np.random.laplace(loc=0, scale=10/1))
    tmp = np.insert(tmp, 4, float(tmp[3]) + np.random.laplace(loc=0, scale=10/1))
    test_sub_feature.append(tmp)

with open("../../../dataset/test/pen_sub_feature_train.csv", "w", encoding="utf-8", newline="") as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(name_feature)
    csv_writer.writerows(train_sub_feature)
    f.close()
with open("../../../dataset/test/pen_sub_feature_validate.csv", "w", encoding="utf-8", newline="") as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(name_feature)
    csv_writer.writerows(validate_sub_feature)
    f.close()
with open("../../../dataset/test/pen_sub_feature_test.csv", "w", encoding="utf-8", newline="") as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(name_feature)
    csv_writer.writerows(test_sub_feature)
    f.close()


# 创造子项的方式
arr_train = arr_train.tolist()
for i in range(len(arr_train)):
    tmp = copy.deepcopy(arr_train[i])
    tmp[1] = float(tmp[1]) + np.random.laplace(loc=0, scale=10/1)
    arr_train.append(tmp)

arr_validate = arr_validate.tolist()
for i in range(len(arr_validate)):
    tmp = copy.deepcopy(arr_validate[i])
    tmp[1] = float(tmp[1]) + np.random.laplace(loc=0, scale=10/1)
    arr_validate.append(tmp)

arr_test = arr_test.tolist()
for i in range(len(arr_test)):
    tmp = copy.deepcopy(arr_test[i])
    tmp[1] = float(tmp[1]) + np.random.laplace(loc=0, scale=10/1)
    arr_test.append(tmp)

for i in range(len(arr_train)):
    tmp = copy.deepcopy(arr_train[i])
    tmp[2] = float(tmp[2]) + np.random.laplace(loc=0, scale=10/1)
    arr_train.append(tmp)

for i in range(len(arr_validate)):
    tmp = copy.deepcopy(arr_validate[i])
    tmp[2] = float(tmp[2]) + np.random.laplace(loc=0, scale=10/1)
    arr_validate.append(tmp)

for i in range(len(arr_test)):
    tmp = copy.deepcopy(arr_test[i])
    tmp[2] = float(tmp[2]) + np.random.laplace(loc=0, scale=10/1)
    arr_test.append(tmp)

arr_train = np.array(arr_train)
arr_validate = np.array(arr_validate)
arr_test = np.array(arr_test)
np.random.shuffle(arr_train)
split = int(arr_train.shape[0]*0.25)
arr_train = arr_train[0:split]
np.random.shuffle(arr_validate)
split = int(arr_validate.shape[0]*0.25)
arr_validate = arr_validate[0:split]
np.random.shuffle(arr_test)
split = int(arr_test.shape[0]*0.25)
arr_test = arr_test[0:split]

with open("../../../dataset/test/pen_sub_item_train.csv", "w", encoding="utf-8", newline="") as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(name)
    csv_writer.writerows(arr_train)
    f.close()
with open("../../../dataset/test/pen_sub_item_validate.csv", "w", encoding="utf-8", newline="") as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(name)
    csv_writer.writerows(arr_validate)
    f.close()

with open("../../../dataset/test/pen_sub_item_test.csv", "w", encoding="utf-8", newline="") as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(name)
    csv_writer.writerows(arr_test)
    f.close()
