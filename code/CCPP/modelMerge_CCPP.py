# 将两种训练方式得到的模型在验证集上得到的预测结果进行融合
# 融合涉及id对齐
# 元学习器选择一个简单的神经网络
import time

import numpy
import keras as K
import tensorflow as tf
import torch
from torch import nn
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBClassifier, XGBRegressor
import numpy as np
from pandas import read_csv
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, balanced_accuracy_score

import torch.utils.data as Data
import hiddenlayer as hl

scale = '10.0_50.0'
# scale = '1.0_5.0'
# scale = '0.1_0.5'
# enc = ''
enc = '_enc'
name_file = 'CCPP'

def get_in_order(X_feature_id, X_id, X, y):
    X_feature_id = X_feature_id.T[0].tolist()
    X_item_id = X_id.T[0].tolist()
    tmp_X = []
    tmp_y = []
    for i in range(len(X_feature_id)):
        tmp_id = X_feature_id[i]
        index_tmp = X_item_id.index(tmp_id)
        tmp_X.append(X[index_tmp])
        tmp_y.append(y[index_tmp])
    return np.array(tmp_X), np.array(tmp_y)

# 读取数据
# 分裂属性的数据
df = read_csv("../../dataset/bias/"+name_file+"/"+name_file+"_sub_feature_train"+enc+scale+".csv")
data = df.values
X_train_feature = data[:, 1:7].astype(int)
X_train_feature_ = data[:, 0:1].astype(int)
y_train_feature = data[:, -1].astype(int)

df = read_csv("../../dataset/bias/"+name_file+"/"+name_file+"_sub_feature_validate"+enc+scale+".csv")
data = df.values
X_validate_feature = data[:, 1:7].astype(int)
X_validate_feature_ = data[:, 0:1].astype(int)
y_validate_feature = data[:, -1].astype(int)


df = read_csv("../../dataset/bias/"+name_file+"/"+name_file+"_sub_feature_test"+enc+scale+".csv")
data = df.values
X_test_feature = data[:, 1:7].astype(int)
X_test_feature_ = data[:, 0:1].astype(int)
y_test_feature = data[:, -1].astype(int)
# 只使用一方数据
df = read_csv("../../dataset/bias/"+name_file+"/"+name_file+"_sub_feature_train"+enc+scale+".csv")
data = df.values
X_train_single_1 = np.concatenate((data[:, 1:2].astype(int), data[:, 3:4].astype(int), data[:, 5:7].astype(int)), axis=1)
X_train_single_2 = np.concatenate((data[:, 1:2].astype(int), data[:, 4:5].astype(int), data[:, 5:7].astype(int)), axis=1)
X_train_single_3 = np.concatenate((data[:, 2:3].astype(int), data[:, 3:4].astype(int), data[:, 5:7].astype(int)), axis=1)
X_train_single_4 = np.concatenate((data[:, 2:3].astype(int), data[:, 4:5].astype(int), data[:, 5:7].astype(int)), axis=1)
X_train_single_ = data[:, 0:1].astype(int)
y_train_single = data[:, -1].astype(int)

df = read_csv("../../dataset/bias/"+name_file+"/"+name_file+"_sub_feature_validate"+enc+scale+".csv")
data = df.values
X_validate_single_1 = np.concatenate((data[:, 1:2].astype(int), data[:, 3:4].astype(int), data[:, 5:7].astype(int)), axis=1)
X_validate_single_2 = np.concatenate((data[:, 1:2].astype(int), data[:, 4:5].astype(int), data[:, 5:7].astype(int)), axis=1)
X_validate_single_3 = np.concatenate((data[:, 2:3].astype(int), data[:, 3:4].astype(int), data[:, 5:7].astype(int)), axis=1)
X_validate_single_4 = np.concatenate((data[:, 2:3].astype(int), data[:, 4:5].astype(int), data[:, 5:7].astype(int)), axis=1)
X_validate_single_ = data[:, 0:1].astype(int)
y_validate_single = data[:, -1].astype(int)


df = read_csv("../../dataset/bias/"+name_file+"/"+name_file+"_sub_feature_test"+enc+scale+".csv")
data = df.values
X_test_single_1 = np.concatenate((data[:, 1:2].astype(int), data[:, 3:4].astype(int), data[:, 5:7].astype(int)), axis=1)
X_test_single_2 = np.concatenate((data[:, 1:2].astype(int), data[:, 4:5].astype(int), data[:, 5:7].astype(int)), axis=1)
X_test_single_3 = np.concatenate((data[:, 2:3].astype(int), data[:, 3:4].astype(int), data[:, 5:7].astype(int)), axis=1)
X_test_single_4 = np.concatenate((data[:, 2:3].astype(int), data[:, 4:5].astype(int), data[:, 5:7].astype(int)), axis=1)
X_test_single_ = data[:, 0:1].astype(int)
y_test_single = data[:, -1].astype(int)
# 分裂条目方式的数据
df = read_csv("../../dataset/bias/"+name_file+"/"+name_file+"_sub_item_train"+enc+scale+".csv")
data = df.values
X_train_item = data[:, 1:5]
X_train_item_ = data[:, 0:1]
y_train_item = data[:, -1]


df = read_csv("../../dataset/bias/"+name_file+"/"+name_file+"_sub_item_validate"+enc+scale+".csv")
data = df.values
X_validate_item = data[:, 1:5]
X_validate_item_ = data[:, 0:1]
y_validate_item = data[:, -1]

df = read_csv("../../dataset/bias/"+name_file+"/"+name_file+"_sub_item_test"+enc+scale+".csv")
data = df.values
X_test_item = data[:, 1:5]
X_test_item_ = data[:, 0:1]
y_test_item = data[:, -1]

n_estimators = 20
max_depth = 4
# 分裂属性的训练
time_start = time.time()  # 开始计时
bst_feature = XGBRegressor(nthread=1, n_estimators=n_estimators, max_depth=max_depth, learning_rate=1, objective='reg:squarederror')
bst_feature.fit(X_train_feature, y_train_feature)
mse_feature1 = mean_squared_error(y_train_feature, bst_feature.predict(X_train_feature))
mse_feature = mean_squared_error(y_validate_feature, bst_feature.predict(X_validate_feature))
mse_feature_test = mean_squared_error(y_test_feature, bst_feature.predict(X_test_feature))
r2_feature = r2_score(y_validate_feature, bst_feature.predict(X_validate_feature))
r2_feature_test = r2_score(y_test_feature, bst_feature.predict(X_test_feature))
accumulation_feature = bst_feature.predict(X_validate_feature, output_margin=True)
accumulation_feature_test = bst_feature.predict(X_test_feature, output_margin=True)
# print("mse feature1:", mse_feature1)
# print("mse feature:", mse_feature)
# print("mse feature test:", mse_feature_test)
# print("r2 feature:", r2_feature)
# print("r2 feature test:", r2_feature_test)
time_end = time.time()  # 结束计时
time_c = time_end - time_start  # 运行所花时间
print('time cost 1', time_c*6, 's')

time_start = time.time()  # 开始计时
# 分裂条目的训练
bst_item = XGBRegressor(nthread=1, n_estimators=n_estimators, max_depth=max_depth, learning_rate=1, objective='reg:squarederror')
# fit model
bst_item.fit(X_train_item, y_train_item)
# make predictions
mse_item = mean_squared_error(y_validate_item, bst_item.predict(X_validate_item))
mse_item_test = mean_squared_error(y_test_item, bst_item.predict(X_test_item))
r2_item = r2_score(y_validate_item, bst_item.predict(X_validate_item))
r2_item_test = r2_score(y_test_item, bst_item.predict(X_test_item))
accumulation_item = bst_item.predict(X_validate_item, output_margin=True)
accumulation_item_test = bst_item.predict(X_test_item, output_margin=True)
# print("mse item:", mse_item)
# print("mse item test:", mse_item_test)
# print("r2 item:", r2_item)
# print("r2 item test:", r2_item_test)
time_end = time.time()  # 结束计时
time_c = time_end - time_start  # 运行所花时间
print('time cost 2', time_c*6, 's')

time_start = time.time()  # 开始计时
# single1
bst_single1 = XGBRegressor(nthread=1, n_estimators=n_estimators, max_depth=max_depth, learning_rate=1, objective='reg:squarederror')
bst_single1.fit(X_train_single_1, y_train_single)
mse_single1 = mean_squared_error(y_validate_single, bst_single1.predict(X_validate_single_1))
mse_single1_test = mean_squared_error(y_test_single, bst_single1.predict(X_test_single_1))
r2_single1 = r2_score(y_validate_single, bst_single1.predict(X_validate_single_1))
r2_single1_test = r2_score(y_test_single, bst_single1.predict(X_test_single_1))
accumulation_single1 = bst_single1.predict(X_validate_single_1, output_margin=True)
accumulation_single1_test = bst_single1.predict(X_test_single_1, output_margin=True)
# print("mse single1:", mse_single1)
# print("mse single1 test:", mse_single1_test)
# print("r2 single1:", r2_single1)
# print("r2 single1 test:", r2_single1_test)
time_end = time.time()  # 结束计时
time_c = time_end - time_start  # 运行所花时间
print('time cost 3', time_c*6, 's')

time_start = time.time()  # 开始计时
# single2
bst_single2 = XGBRegressor(nthread=1, n_estimators=n_estimators, max_depth=max_depth, learning_rate=1, objective='reg:squarederror')
bst_single2.fit(X_train_single_2, y_train_single)
mse_single2 = mean_squared_error(y_validate_single, bst_single2.predict(X_validate_single_2))
mse_single2_test = mean_squared_error(y_test_single, bst_single2.predict(X_test_single_2))
r2_single2 = r2_score(y_validate_single, bst_single2.predict(X_validate_single_2))
r2_single2_test = r2_score(y_test_single, bst_single2.predict(X_test_single_2))
accumulation_single2 = bst_single2.predict(X_validate_single_2, output_margin=True)
accumulation_single2_test = bst_single2.predict(X_test_single_2, output_margin=True)
# print("mse single2:", mse_single2)
# print("mse single2 test:", mse_single2_test)
# print("r2 single2:", r2_single2)
# print("r2 single2 test:", r2_single2_test)
time_end = time.time()  # 结束计时
time_c = time_end - time_start  # 运行所花时间
print('time cost 4', time_c*6, 's')

time_start = time.time()  # 开始计时
# single3
bst_single3 = XGBRegressor(nthread=1, n_estimators=n_estimators, max_depth=max_depth, learning_rate=1, objective='reg:squarederror')
bst_single3.fit(X_train_single_3, y_train_single)
mse_single3 = mean_squared_error(y_validate_single, bst_single3.predict(X_validate_single_3))
mse_single3_test = mean_squared_error(y_test_single, bst_single3.predict(X_test_single_3))
r2_single3 = r2_score(y_validate_single, bst_single3.predict(X_validate_single_3))
r2_single3_test = r2_score(y_test_single, bst_single3.predict(X_test_single_3))
accumulation_single3 = bst_single3.predict(X_validate_single_3, output_margin=True)
accumulation_single3_test = bst_single3.predict(X_test_single_3, output_margin=True)
# print("mse single3:", mse_single3)
# print("mse single3 test:", mse_single3_test)
# print("r2 single3:", r2_single3)
# print("r2 single3 test:", r2_single3_test)
time_end = time.time()  # 结束计时
time_c = time_end - time_start  # 运行所花时间
print('time cost 5', time_c*6, 's')

time_start = time.time()  # 开始计时
# single4
bst_single4 = XGBRegressor(nthread=1, n_estimators=n_estimators, max_depth=max_depth, learning_rate=1, objective='reg:squarederror')
bst_single4.fit(X_train_single_4, y_train_single)
mse_single4 = mean_squared_error(y_validate_single, bst_single4.predict(X_validate_single_4))
mse_single4_test = mean_squared_error(y_test_single, bst_single4.predict(X_test_single_4))
r2_single4 = r2_score(y_validate_single, bst_single4.predict(X_validate_single_4))
r2_single4_test = r2_score(y_test_single, bst_single4.predict(X_test_single_4))
accumulation_single4 = bst_single4.predict(X_validate_single_4, output_margin=True)
accumulation_single4_test = bst_single4.predict(X_test_single_4, output_margin=True)
# print("mse single4:", mse_single4)
# print("mse single4 test:", mse_single4_test)
# print("r2 single4:", r2_single4)
# print("r2 single4 test:", r2_single4_test)
time_end = time.time()  # 结束计时
time_c = time_end - time_start  # 运行所花时间
print('time cost 6', time_c*6, 's')
# 将其他数据集顺序和feature顺序同步
X_feature_train = accumulation_feature
y_feature_train = y_validate_feature
X_item_train, y_item_train = get_in_order(X_validate_feature_, X_validate_item_, accumulation_item, y_validate_item)
X_single1_train, y_single1_train = get_in_order(X_validate_feature_, X_validate_single_, accumulation_single1, y_validate_single)
X_single2_train, y_single2_train = get_in_order(X_validate_feature_, X_validate_single_, accumulation_single2, y_validate_single)
X_single3_train, y_single3_train = get_in_order(X_validate_feature_, X_validate_single_, accumulation_single3, y_validate_single)
X_single4_train, y_single4_train = get_in_order(X_validate_feature_, X_validate_single_, accumulation_single4, y_validate_single)


X_feature_test = accumulation_feature_test
y_feature_test = y_test_feature
X_item_test, y_item_test = get_in_order(X_test_feature_, X_test_item_, accumulation_item_test, y_test_item)
X_single1_test, y_single1_test = get_in_order(X_test_feature_, X_test_single_, accumulation_single1_test, y_test_single)
X_single2_test, y_single2_test = get_in_order(X_test_feature_, X_test_single_, accumulation_single2_test, y_test_single)
X_single3_test, y_single3_test = get_in_order(X_test_feature_, X_test_single_, accumulation_single3_test, y_test_single)
X_single4_test, y_single4_test = get_in_order(X_test_feature_, X_test_single_, accumulation_single4_test, y_test_single)




data_train_second = np.concatenate(([X_feature_train], [X_item_train], [X_single1_train], [X_single2_train], [X_single3_train], [X_single4_train]), axis=0).T
target_train_second = np.array(y_feature_train)
data_val_second = np.concatenate(([X_feature_test], [X_item_test], [X_single1_test], [X_single2_test], [X_single3_test], [X_single4_test]), axis=0).T
target_val_second = np.array(y_feature_test)



b_size = 512
max_epochs = 30
num_iteration = 1
time_start = time.time()  # 开始计时
for ep in range(29, max_epochs):
    ba_sum = 0
    acc_sum = 0
    for i in range(num_iteration):
        # 2. 定义模型
        init = K.initializers.glorot_uniform(seed=1)
        simple_adam = K.optimizers.Adam()
        model = K.models.Sequential()
        model.add(K.layers.Dense(units=18, input_dim=6, activation='relu'))
        model.add(K.layers.Dense(units=12, input_dim=18, activation='relu'))
        model.add(K.layers.Dense(units=1))
        model.compile(loss='mse', optimizer=simple_adam, metrics=['mse'])
        # print("iteration:", i)
        # print("Starting training ")
        h = model.fit(data_train_second, target_train_second, batch_size=b_size, epochs=ep + 1, shuffle=True, verbose=1)
        # print("Training finished \n")

        # 4. 评估模型
        # pred = model.predict(data_val_second)
        # mse_val = mean_squared_error(target_val_second, pred)
        # r2_val = r2_score(target_val_second, pred)
        # print("mse validate: ", mse_val)
        # print("r2 validate: ", r2_val)
    # mean_ba = ba_sum/num_iteration
    # mean_acc = acc_sum / num_iteration
    # print("epochs:", ep + 1, "mean balanced acc:", mean_ba)
    # print("epochs:", ep + 1, "mean acc:", mean_acc)
time_end = time.time()  # 结束计时
time_c = time_end - time_start  # 运行所花时间
print('time cost 7', time_c, 's')