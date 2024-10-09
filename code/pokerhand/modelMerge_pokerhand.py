# 将两种训练方式得到的模型在验证集上得到的预测结果进行融合
# 融合涉及id对齐
# 元学习器选择一个简单的神经网络
import numpy
import keras as K
import tensorflow as tf
import torch
from torch import nn
from tqdm import tqdm
from xgboost import XGBClassifier
import numpy as np
from pandas import read_csv
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, balanced_accuracy_score

import torch.utils.data as Data
import hiddenlayer as hl

# scale = '0.1_0.1'
# scale = '1.0_1.0'
scale = '10.0_10.0'
enc = ''
# enc = '_enc'

# 读取数据
# 分裂属性的数据
df = read_csv("../../dataset/bias/poker/poker_sub_feature_train"+enc+scale+".csv")
data = df.values
X_train_feature = data[:, 1:13].astype(int)
X_train_feature_ = data[:, 0:1].astype(int)
y_train_feature = data[:, -1].astype(int)

df = read_csv("../../dataset/bias/poker/poker_sub_feature_validate"+enc+scale+".csv")
data = df.values
X_validate_feature = data[:, 1:13].astype(int)
X_validate_feature_ = data[:, 0:1].astype(int)
y_validate_feature = data[:, -1].astype(int)


df = read_csv("../../dataset/bias/poker/poker_sub_feature_test"+enc+scale+".csv")
data = df.values
X_test_feature = data[:, 1:13].astype(int)
X_test_feature_ = data[:, 0:1].astype(int)
y_test_feature = data[:, -1].astype(int)
# 只使用一方数据
df = read_csv("../../dataset/bias/poker/poker_sub_feature_train"+enc+scale+".csv")
data = df.values
X_train_single_1 = np.concatenate((data[:, 1:2].astype(int), data[:, 3:4].astype(int), data[:, 5:13].astype(int)), axis=1)
X_train_single_2 = np.concatenate((data[:, 1:2].astype(int), data[:, 4:5].astype(int), data[:, 5:13].astype(int)), axis=1)
X_train_single_3 = np.concatenate((data[:, 2:3].astype(int), data[:, 3:4].astype(int), data[:, 5:13].astype(int)), axis=1)
X_train_single_4 = np.concatenate((data[:, 2:3].astype(int), data[:, 4:5].astype(int), data[:, 5:13].astype(int)), axis=1)
X_train_single_ = data[:, 0:1].astype(int)
y_train_single = data[:, -1].astype(int)

df = read_csv("../../dataset/bias/poker/poker_sub_feature_validate"+enc+scale+".csv")
data = df.values
X_validate_single_1 = np.concatenate((data[:, 1:2].astype(int), data[:, 3:4].astype(int), data[:, 5:13].astype(int)), axis=1)
X_validate_single_2 = np.concatenate((data[:, 1:2].astype(int), data[:, 4:5].astype(int), data[:, 5:13].astype(int)), axis=1)
X_validate_single_3 = np.concatenate((data[:, 2:3].astype(int), data[:, 3:4].astype(int), data[:, 5:13].astype(int)), axis=1)
X_validate_single_4 = np.concatenate((data[:, 2:3].astype(int), data[:, 4:5].astype(int), data[:, 5:13].astype(int)), axis=1)
X_validate_single_ = data[:, 0:1].astype(int)
y_validate_single = data[:, -1].astype(int)


df = read_csv("../../dataset/bias/poker/poker_sub_feature_test"+enc+scale+".csv")
data = df.values
X_test_single_1 = np.concatenate((data[:, 1:2].astype(int), data[:, 3:4].astype(int), data[:, 5:13].astype(int)), axis=1)
X_test_single_2 = np.concatenate((data[:, 1:2].astype(int), data[:, 4:5].astype(int), data[:, 5:13].astype(int)), axis=1)
X_test_single_3 = np.concatenate((data[:, 2:3].astype(int), data[:, 3:4].astype(int), data[:, 5:13].astype(int)), axis=1)
X_test_single_4 = np.concatenate((data[:, 2:3].astype(int), data[:, 4:5].astype(int), data[:, 5:13].astype(int)), axis=1)
X_test_single_ = data[:, 0:1].astype(int)
y_test_single = data[:, -1].astype(int)
# 分裂条目方式的数据
df = read_csv("../../dataset/bias/poker/poker_sub_item_train"+enc+scale+".csv")
data = df.values
X_train_item = data[:, 1:11]
X_train_item_ = data[:, 0:1]
y_train_item = data[:, -1]


df = read_csv("../../dataset/bias/poker/poker_sub_item_validate"+enc+scale+".csv")
data = df.values
X_validate_item = data[:, 1:11]
X_validate_item_ = data[:, 0:1]
y_validate_item = data[:, -1]

df = read_csv("../../dataset/bias/poker/poker_sub_item_test"+enc+scale+".csv")
data = df.values
X_test_item = data[:, 1:11]
X_test_item_ = data[:, 0:1]
y_test_item = data[:, -1]

# 分裂属性的训练
bst_feature = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=1, objective='multi:softprob')
bst_feature.fit(X_train_feature, y_train_feature)
acc_feature = accuracy_score(y_validate_feature, bst_feature.predict(X_validate_feature))
b_acc_feature = balanced_accuracy_score(y_validate_feature, bst_feature.predict(X_validate_feature))
accumulation_feature = bst_feature.predict(X_validate_feature, output_margin=True)
print("acc feature:", acc_feature)
print("acc balanced feature:", b_acc_feature)
#
# 分裂条目的训练
bst_item = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=1, objective='multi:softprob')
# fit model
bst_item.fit(X_train_item, y_train_item)
# make predictions
acc_item = accuracy_score(y_validate_item, bst_item.predict(X_validate_item))
b_acc_item = balanced_accuracy_score(y_validate_item, bst_item.predict(X_validate_item))
accumulation_item = bst_item.predict(X_validate_item, output_margin=True)
print("acc item:", acc_item)
print("acc balanced item:", b_acc_item)
# single1
bst_item = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=1, objective='multi:softprob')
# fit model
bst_item.fit(X_train_single_1, y_train_single)
# make predictions
acc_single1 = accuracy_score(y_validate_single, bst_item.predict(X_validate_single_1))
b_acc_single1 = balanced_accuracy_score(y_validate_single, bst_item.predict(X_validate_single_1))
accumulation_single1 = bst_item.predict(X_validate_single_1, output_margin=True)
print("acc single1:", acc_single1)
print("acc balanced single1:", b_acc_single1)

# single2
bst_item = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=1, objective='multi:softprob')
# fit model
bst_item.fit(X_train_single_2, y_train_single)
# make predictions
acc_single2 = accuracy_score(y_validate_single, bst_item.predict(X_validate_single_2))
b_acc_single2 = balanced_accuracy_score(y_validate_single, bst_item.predict(X_validate_single_2))
accumulation_single2 = bst_item.predict(X_validate_single_2, output_margin=True)
print("acc single2:", acc_single2)
print("acc balanced single2:", b_acc_single2)

# single3
bst_item = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=1, objective='multi:softprob')
# fit model
bst_item.fit(X_train_single_3, y_train_single)
# make predictions
acc_single3 = accuracy_score(y_validate_single, bst_item.predict(X_validate_single_3))
b_acc_single3 = balanced_accuracy_score(y_validate_single, bst_item.predict(X_validate_single_3))
accumulation_single3 = bst_item.predict(X_validate_single_3, output_margin=True)
print("acc single3:", acc_single3)
print("acc balanced single3:", b_acc_single3)

# single4
bst_item = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=1, objective='multi:softprob')
# fit model
bst_item.fit(X_train_single_4, y_train_single)
# make predictions
acc_single4 = accuracy_score(y_validate_single, bst_item.predict(X_validate_single_4))
b_acc_single4 = balanced_accuracy_score(y_validate_single, bst_item.predict(X_validate_single_4))
accumulation_single4 = bst_item.predict(X_validate_single_4, output_margin=True)
print("acc single4:", acc_single4)
print("acc balanced single4:", b_acc_single4)
# 第二层训练的训练集
dict_tmp = {}
for i in range(accumulation_feature.shape[0]):
    dict_tmp.update({int(X_validate_feature_[i][0]): accumulation_feature[i]})
data_train_second = []
label_train_second = []
# for i in range(accumulation_single3.shape[0]):
#     # id
#     arr1 = dict_tmp[int(X_validate_single_[i][0])]
#     arr2 = accumulation_single3[i]
#     tmp = numpy.concatenate((arr1, arr2), axis=0)
#     data_train_second.append(tmp)
#     label_train_second.append(y_validate_single[i])
for i in range(accumulation_item.shape[0]):
    # id
    arr1 = dict_tmp[int(X_validate_item_[i][0])]
    arr2 = accumulation_item[i]
    tmp = numpy.concatenate((arr1, arr2), axis=0)
    data_train_second.append(tmp)
    label_train_second.append(y_validate_item[i])
# 第二层的测试集
# accumulation_feature_test_second = bst_feature.predict(X_test_feature, output_margin=True)
# accumulation_single_test_second = bst_item.predict(X_test_single_3, output_margin=True)
# dict_tmp = {}
# for i in range(accumulation_feature_test_second.shape[0]):
#     dict_tmp.update({int(X_test_feature_[i][0]): accumulation_feature_test_second[i]})
# data_val_second = []
# label_val_second = []
# for i in range(accumulation_single_test_second.shape[0]):
#     # id
#     arr1 = dict_tmp[int(X_test_single_[i][0])]
#     arr2 = accumulation_single_test_second[i]
#     tmp = numpy.concatenate((arr1, arr2), axis=0)
#     data_val_second.append(tmp)
#     label_val_second.append(y_test_item[i])
accumulation_feature_test_second = bst_feature.predict(X_test_feature, output_margin=True)
accumulation_item_test_second = bst_item.predict(X_test_item, output_margin=True)
dict_tmp = {}
for i in range(accumulation_feature_test_second.shape[0]):
    dict_tmp.update({int(X_test_feature_[i][0]): accumulation_feature_test_second[i]})
data_val_second = []
label_val_second = []
for i in range(accumulation_item_test_second.shape[0]):
    # id
    arr1 = dict_tmp[int(X_test_item_[i][0])]
    arr2 = accumulation_item_test_second[i]
    tmp = numpy.concatenate((arr1, arr2), axis=0)
    data_val_second.append(tmp)
    label_val_second.append(y_test_item[i])

data_train_second = np.array(data_train_second)
label_train_second = np.array(label_train_second)
data_val_second = np.array(data_val_second)
label_val_second = np.array(label_val_second)



b_size = 32
max_epochs = 10
num_iteration = 1
for ep in range(max_epochs):
    ba_sum = 0
    acc_sum = 0
    for i in range(num_iteration):
        # 2. 定义模型
        init = K.initializers.glorot_uniform(seed=1)
        simple_adam = K.optimizers.Adam()
        model = K.models.Sequential()
        model.add(K.layers.Dense(units=15, input_dim=20, kernel_initializer=init, activation='relu'))
        # model.add(K.layers.Dense(units=15, kernel_initializer=init, activation='relu'))
        model.add(K.layers.Dense(units=10, kernel_initializer=init, activation='softmax'))
        model.compile(loss='sparse_categorical_crossentropy', optimizer=simple_adam, metrics=['accuracy'])
        print("iteration:", i)
        print("Starting training ")
        h = model.fit(data_train_second, label_train_second, batch_size=b_size, epochs=ep+1, shuffle=True, verbose=1)
        print("Training finished \n")

        # 4. 评估模型
        eval = model.evaluate(data_val_second, label_val_second, verbose=0)
        print("Evaluation on test data: loss = %0.6f accuracy = %0.2f%% \n" \
              % (eval[0], eval[1] * 100))
        ba = balanced_accuracy_score(label_val_second, numpy.argmax(model.predict(data_val_second), axis=1))
        print("balanced acc:", ba)
        ba_sum = ba_sum + ba
        acc = accuracy_score(label_val_second, numpy.argmax(model.predict(data_val_second), axis=1))
        print("acc:", acc)
        acc_sum = acc_sum + acc
    mean_ba = ba_sum/num_iteration
    mean_acc = acc_sum / num_iteration
    print("epochs:", ep + 1, "mean balanced acc:", mean_ba)
    print("epochs:", ep + 1, "mean acc:", mean_acc)
