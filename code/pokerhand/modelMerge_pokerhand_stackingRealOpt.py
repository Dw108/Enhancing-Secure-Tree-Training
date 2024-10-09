# 将两种训练方式得到的模型在验证集上得到的预测结果进行融合
# 融合涉及id对齐
# 元学习器选择一个简单的神经网络
import numpy
import keras as K
import copy
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier
import numpy as np
from pandas import read_csv
from numpy import sort
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import script.network
import torch.utils.data as Data
import hiddenlayer as hl
# scale = '1000.0_1000.0'
# scale = '100.0_100.0'
scale = '10.0_10.0'

# enc = ''
enc = '_enc'
name_file = 'poker'

dp_scale = 0.0001

def dp_noise(data_tmp, scale_tmp):
    mean_value = np.average(data_tmp, axis=0)
    for i in range(data_tmp.shape[1]):
        true_scale = mean_value[i] * scale_tmp
        for j in range(data_tmp.shape[0]):
            data_tmp[j][i] = data_tmp[j][i] + np.random.laplace(loc=0, scale=true_scale)
    return data_tmp


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
df = read_csv("../../dataset/bias/"+name_file+"/"+name_file+"_sub_feature_train" + enc + scale + ".csv")

data = df.values
X_train_feature = data[:, 1:13].astype(float)
X_train_feature = dp_noise(X_train_feature, dp_scale)
X_train_feature_ = data[:, 0:1].astype(float)
y_train_feature = data[:, -1].astype(float)

df = read_csv("../../dataset/bias/"+name_file+"/"+name_file+"_sub_feature_validate" + enc + scale + ".csv")
data = df.values
X_validate_feature = data[:, 1:13].astype(float)
X_validate_feature = dp_noise(X_validate_feature, dp_scale)
X_validate_feature_ = data[:, 0:1].astype(float)
y_validate_feature = data[:, -1].astype(float)

df = read_csv("../../dataset/bias/"+name_file+"/"+name_file+"_sub_feature_test" + enc + scale + ".csv")
data = df.values
X_test_feature = data[:, 1:13].astype(float)
X_test_feature = dp_noise(X_test_feature, dp_scale)
X_test_feature_ = data[:, 0:1].astype(float)
y_test_feature = data[:, -1].astype(float)

# 分裂条目方式的数据
df = read_csv("../../dataset/bias/"+name_file+"/"+name_file+"_sub_item_train" + enc + scale + ".csv")
data = df.values
X_train_item = data[:, 1:11]
X_train_item = dp_noise(X_train_item, dp_scale)
X_train_item_ = data[:, 0:1]
y_train_item = data[:, -1]

df = read_csv("../../dataset/bias/"+name_file+"/"+name_file+"_sub_item_validate" + enc + scale + ".csv")
data = df.values
X_validate_item = data[:, 1:11]
X_validate_item = dp_noise(X_validate_item, dp_scale)
X_validate_item_ = data[:, 0:1]
y_validate_item = data[:, -1]

df = read_csv("../../dataset/bias/"+name_file+"/"+name_file+"_sub_item_test" + enc + scale + ".csv")
data = df.values
X_test_item = data[:, 1:11]
X_test_item = dp_noise(X_test_item, dp_scale)
X_test_item_ = data[:, 0:1]
y_test_item = data[:, -1]

X_feature = np.concatenate((X_train_feature, X_validate_feature, X_test_feature))
X_feature_id = np.concatenate((X_train_feature_, X_validate_feature_, X_test_feature_))
y_feature = np.concatenate((y_train_feature, y_validate_feature, y_test_feature))

X_item = np.concatenate((X_train_item, X_validate_item, X_test_item))
X_item_id = np.concatenate((X_train_item_, X_validate_item_, X_test_item_))
y_item = np.concatenate((y_train_item, y_validate_item, y_test_item))

# X_single1 = np.concatenate((X_train_single_1, X_validate_single_1, X_test_single_1))
# X_single1_id = np.concatenate((X_train_single1_, X_validate_single1_, X_test_single1_))
# y_single1 = np.concatenate((y_train_single1_, y_validate_single1_, y_test_single1_))
#
# X_single2 = np.concatenate((X_train_single_2, X_validate_single_2, X_test_single_2))
# X_single2_id = np.concatenate((X_train_single2_, X_validate_single2_, X_test_single2_))
# y_single2 = np.concatenate((y_train_single2_, y_validate_single2_, y_test_single2_))
#
# X_single3 = np.concatenate((X_train_single_3, X_validate_single_3, X_test_single_3))
# X_single3_id = np.concatenate((X_train_single3_, X_validate_single3_, X_test_single3_))
# y_single3 = np.concatenate((y_train_single3_, y_validate_single3_, y_test_single3_))
#
# X_single4 = np.concatenate((X_train_single_4, X_validate_single_4, X_test_single_4))
# X_single4_id = np.concatenate((X_train_single4_, X_validate_single4_, X_test_single4_))
# y_single4 = np.concatenate((y_train_single4_, y_validate_single4_, y_test_single4_))
# 将其他数据集顺序和feature顺序同步
X_item, y_item = get_in_order(X_feature_id, X_item_id, X_item, y_item)
# 共有6个模型，将数据均分为7份，其中6份用于K折验证，1份用于测试
share_size = int(X_feature_id.shape[0] / 4)

X_feature_train = X_feature[0:share_size * 2, :]
X_feature_val = X_feature[share_size * 2:share_size * 3, :]
X_feature_test = X_feature[share_size * 3:, :]
y_feature_train = y_feature[0:share_size * 2]
y_feature_val = y_feature[share_size * 2:share_size * 3]
y_feature_test = y_feature[share_size * 3:]

X_feature_second_train = X_feature[0:share_size * 3, :]
y_feature_second_train = y_feature[0:share_size * 3]


X_item_train = np.concatenate((X_item[0:share_size * 1, :], X_item[share_size * 2:share_size * 3, :]))
X_item_val = X_item[share_size * 1:share_size * 2, :]
X_item_test = X_item[share_size * 3:, :]
y_item_train = np.concatenate((y_item[0:share_size * 1], y_item[share_size * 2:share_size * 3]))
y_item_val = y_item[share_size * 1:share_size * 2]
y_item_test = y_item[share_size * 3:]

X_item_second_train = X_item[0:share_size * 3, :]
y_item_second_train = y_item[0:share_size * 3]



X_select_train = X_feature[share_size * 1:share_size * 3, :]
X_select_val = X_feature[share_size * 0:share_size * 1, :]
X_select_test = X_feature[share_size * 3:, :]
y_select_train = y_feature[share_size * 1:share_size * 3]
y_select_val = y_feature[share_size * 0:share_size * 1]
y_select_test = y_feature[share_size * 3:]

X_select_second_train = X_feature[0:share_size * 3, :]
y_select_second_train = y_feature[0:share_size * 3]


# X_single1_train = np.concatenate((X_single1[0:share_size * 3, :], X_single1[share_size * 4:share_size * 6, :]))
# X_single1_val = X_single1[share_size * 3:share_size * 4, :]
# X_single1_test = X_single1[share_size * 6:, :]
# y_single1_train = np.concatenate((y_single1[0:share_size * 3], y_single1[share_size * 4:share_size * 6]))
# y_single1_val = y_single1[share_size * 3:share_size * 4]
# y_single1_test = y_single1[share_size * 6:]
#
# X_single2_train = np.concatenate((X_single2[0:share_size * 2, :], X_single2[share_size * 3:share_size * 6, :]))
# X_single2_val = X_single2[share_size * 2:share_size * 3, :]
# X_single2_test = X_single2[share_size * 6:, :]
# y_single2_train = np.concatenate((y_single2[0:share_size * 2], y_single2[share_size * 3:share_size * 6]))
# y_single2_val = y_single2[share_size * 2:share_size * 3]
# y_single2_test = y_single2[share_size * 6:]
#
# X_single3_train = np.concatenate((X_single3[0:share_size * 1, :], X_single3[share_size * 2:share_size * 6, :]))
# X_single3_val = X_single3[share_size * 1:share_size * 2, :]
# X_single3_test = X_single3[share_size * 6:, :]
# y_single3_train = np.concatenate((y_single3[0:share_size * 1], y_single3[share_size * 2:share_size * 6]))
# y_single3_val = y_single3[share_size * 1:share_size * 2]
# y_single3_test = y_single3[share_size * 6:]
#
# X_single4_train = X_single4[share_size * 1:share_size * 6, :]
# X_single4_val = X_single4[share_size * 0:share_size * 1, :]
# X_single4_test = X_single4[share_size * 6:, :]
# y_single4_train = y_single4[share_size * 1:share_size * 6]
# y_single4_val = y_single4[share_size * 0:share_size * 1]
# y_single4_test = y_single4[share_size * 6:]

# tmp_1 = []
# for i in range(X_item_id.shape[0]):
#     tmp_1.append(X_item_id[i][0])
# tmp_1 = set(tmp_1)
# num = 0
# for i in range(X_feature_id.shape[0]):
#     if X_feature_id[i][0] not in tmp_1:
#         num = num+1
# print(num)

n_estimators = 20
max_depth = 3
# 分裂属性的训练
bst_feature = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=1, objective='multi:softprob')
bst_feature.fit(X_feature_train, y_feature_train)
acc_feature = accuracy_score(y_feature_val, bst_feature.predict(X_feature_val))
acc_feature1 = accuracy_score(y_feature_train, bst_feature.predict(X_feature_train))
acc_feature_test = accuracy_score(y_feature_test, bst_feature.predict(X_feature_test))
b_acc_feature = balanced_accuracy_score(y_feature_val, bst_feature.predict(X_feature_val))
b_acc_feature_test = balanced_accuracy_score(y_feature_test, bst_feature.predict(X_feature_test))
accumulation_feature_second_train = bst_feature.predict(X_feature_second_train, output_margin=True)
accumulation_feature_second_test = bst_feature.predict(X_feature_test, output_margin=True)
print("acc feature1:", acc_feature1)
print("acc feature:", acc_feature)
print("acc feature test:", acc_feature_test)
print("acc balanced feature:", b_acc_feature)
print("acc balanced feature test:", b_acc_feature_test)
#
# 分裂条目的训练
bst_item = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=1, objective='multi:softprob')
bst_item.fit(X_item_train, y_item_train)
acc_item = accuracy_score(y_item_val, bst_item.predict(X_item_val))
acc_item_test = accuracy_score(y_item_test, bst_item.predict(X_item_test))
b_acc_item = balanced_accuracy_score(y_item_val, bst_item.predict(X_item_val))
b_acc_item_test = balanced_accuracy_score(y_item_test, bst_item.predict(X_item_test))
accumulation_item_second_train = bst_item.predict(X_item_second_train, output_margin=True)
accumulation_item_second_test = bst_item.predict(X_item_test, output_margin=True)
print("acc item:", acc_item)
print("acc item test:", acc_item_test)
print("acc balanced item:", b_acc_item)
print("acc balanced test:", b_acc_item_test)

# fit model on all training data
model = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=1, objective='multi:softprob')
model.fit(X_select_train, y_select_train)
# make predictions for test data and evaluate
y_pred = model.predict(X_select_val)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_select_val, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
# Fit model using each importance as a threshold
thresholds = sort(model.feature_importances_)
max_acc = 0
best_model = 0
best_X_train = 0
best_X_val = 0
best_X_test = 0
for thresh in thresholds:
    # select features using threshold
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    select_X_train = selection.transform(X_select_train)
    # train model
    selection_model = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=1, objective='multi:softprob')
    selection_model.fit(select_X_train, y_select_train)
    # eval model
    select_X_val = selection.transform(X_select_val)
    y_pred = selection_model.predict(select_X_val)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_select_val, predictions)
    if accuracy > max_acc:
        max_acc = accuracy
        best_model = copy.deepcopy(selection_model)
        best_X_train = copy.deepcopy(selection.transform(X_select_second_train))
        best_X_val = copy.deepcopy(selection.transform(X_select_val))
        best_X_test = copy.deepcopy(selection.transform(X_select_test))
    print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy * 100.0))

y_pred = best_model.predict(best_X_val)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_select_val, predictions)
print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, best_X_val.shape[1], accuracy * 100.0))
acc_select = accuracy_score(y_select_val, best_model.predict(best_X_val))
acc_select_test = accuracy_score(y_select_test, best_model.predict(best_X_test))
b_acc_select = balanced_accuracy_score(y_select_val, best_model.predict(best_X_val))
b_acc_select_test = balanced_accuracy_score(y_select_test, best_model.predict(best_X_test))

print("acc select:", acc_select)
print("acc select test:", acc_select_test)
print("acc balanced select:", b_acc_select)
print("acc balanced select test:", b_acc_select_test)

accumulation_select_second_train = best_model.predict(best_X_train, output_margin=True)
accumulation_select_second_test = best_model.predict(best_X_test, output_margin=True)


# # single1
# bst_single1 = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=1, objective='multi:softprob')
# bst_single1.fit(X_single1_train, y_single1_train)
# acc_single1 = accuracy_score(y_single1_val, bst_single1.predict(X_single1_val))
# acc_single1_test = accuracy_score(y_single1_test, bst_single1.predict(X_single1_test))
# b_acc_single1 = balanced_accuracy_score(y_single1_val, bst_single1.predict(X_single1_val))
# b_acc_single1_test = balanced_accuracy_score(y_single1_test, bst_single1.predict(X_single1_test))
# accumulation_single1_second_train = bst_single1.predict(X_single1_val, output_margin=True)
# accumulation_single1_second_test = bst_single1.predict(X_single1_test, output_margin=True)
# print("acc single1:", acc_single1)
# print("acc single1 test:", acc_single1_test)
# print("acc balanced single1:", b_acc_single1)
# print("acc balanced single1 test:", b_acc_single1_test)
#
# # single2
# bst_single2 = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=1, objective='multi:softprob')
# bst_single2.fit(X_single2_train, y_single2_train)
# acc_single2 = accuracy_score(y_single2_val, bst_single2.predict(X_single2_val))
# b_acc_single2 = balanced_accuracy_score(y_single2_val, bst_single2.predict(X_single2_val))
# b_acc_single2_test = balanced_accuracy_score(y_single2_test, bst_single2.predict(X_single2_test))
# acc_single2_test = accuracy_score(y_single2_test, bst_single2.predict(X_single2_test))
# accumulation_single2_second_train = bst_single2.predict(X_single2_val, output_margin=True)
# accumulation_single2_second_test = bst_single2.predict(X_single2_test, output_margin=True)
# print("acc single2:", acc_single2)
# print("acc single2 test:", acc_single2_test)
# print("acc balanced single2:", b_acc_single2)
# print("acc balanced single2 test:", b_acc_single2_test)
#
# # single3
# bst_single3 = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=1, objective='multi:softprob')
# bst_single3.fit(X_single3_train, y_single3_train)
# acc_single3 = accuracy_score(y_single3_val, bst_single3.predict(X_single3_val))
# acc_single3_test = accuracy_score(y_single3_test, bst_single3.predict(X_single3_test))
# b_acc_single3 = balanced_accuracy_score(y_single3_val, bst_single3.predict(X_single3_val))
# b_acc_single3_test = balanced_accuracy_score(y_single3_test, bst_single3.predict(X_single3_test))
# accumulation_single3_second_train = bst_single3.predict(X_single3_val, output_margin=True)
# accumulation_single3_second_test = bst_single3.predict(X_single3_test, output_margin=True)
# print("acc single3:", acc_single3)
# print("acc single3 test:", acc_single3_test)
# print("acc balanced single3:", b_acc_single3)
# print("acc balanced single3 test:", b_acc_single3_test)
#
# # single4
# bst_single4 = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=1, objective='multi:softprob')
# bst_single4.fit(X_single4_train, y_single4_train)
# acc_single4 = accuracy_score(y_single4_val, bst_single4.predict(X_single4_val))
# acc_single4_test = accuracy_score(y_single4_test, bst_single4.predict(X_single4_test))
# b_acc_single4 = balanced_accuracy_score(y_single4_val, bst_single4.predict(X_single4_val))
# b_acc_single4_test = balanced_accuracy_score(y_single4_test, bst_single4.predict(X_single4_test))
# accumulation_single4_second_train = bst_single4.predict(X_single4_val, output_margin=True)
# accumulation_single4_second_test = bst_single4.predict(X_single4_test, output_margin=True)

# print("acc single4:", acc_single4)
# print("acc single4 test:", acc_single4_test)
# print("acc balanced single4:", b_acc_single4)
# print("acc balanced single4 test:", b_acc_single4_test)
# 拼接前做归一化
data_train_second = np.concatenate((accumulation_feature_second_train, accumulation_item_second_train,
                                    accumulation_select_second_train), axis=1)
label_train_second = y_select_second_train

data_val_second0 = np.array(accumulation_feature_second_test)
data_val_second1 = np.array(accumulation_item_second_test)
data_val_second2 = np.array(accumulation_select_second_test)
data_val_second = np.concatenate((data_val_second0, data_val_second1,
                                    data_val_second2), axis=1)
label_val = y_feature_test


b_size = 256
max_epochs = 30
num_iteration = 10
for ep in range(max_epochs - 1, max_epochs):
    ba_sum = 0
    acc_sum = 0
    acc_max = 0
    for i in range(num_iteration):
        # 2. 定义模型
        init = K.initializers.glorot_uniform(seed=1)
        simple_adam = K.optimizers.Adam()
        model = K.models.Sequential()
        # units 是输出大小
        model.add(K.layers.Dense(units=20, input_dim=30, activation='relu'))
        # model.add(K.layers.Dense(units=10, kernel_initializer=init, activation='relu'))
        model.add(K.layers.Dense(units=15, activation='relu'))
        model.add(K.layers.Dense(units=10, activation='softmax'))
        model.compile(loss='sparse_categorical_crossentropy', optimizer=simple_adam, metrics=['accuracy'])
        print("iteration:", i)
        print("Starting training ")
        h = model.fit(data_train_second, label_train_second, batch_size=b_size, epochs=ep + 1, shuffle=True, verbose=1)
        print("Training finished \n")

        pred = model.predict(data_val_second)
        acc = accuracy_score(label_val, numpy.argmax(pred, axis=1))
        b_acc = balanced_accuracy_score(label_val, numpy.argmax(pred, axis=1))
        if acc>acc_max:
            acc_max = acc
        print('acc', acc)
        print('b_acc', b_acc)
        acc_sum = acc_sum + acc
    # mean_ba = ba_sum / num_iteration
    mean_acc = acc_sum / num_iteration
    # print("epochs:", ep + 1, "mean balanced acc:", mean_ba)
    print("epochs:", ep + 1, "mean acc:", mean_acc)
    print(acc_max)