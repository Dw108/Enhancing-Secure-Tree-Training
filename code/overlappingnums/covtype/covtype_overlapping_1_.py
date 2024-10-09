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
import script.network
import torch.utils.data as Data
import hiddenlayer as hl

# enc = ''
enc = '_enc'
name_file = 'covtype'

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
df = read_csv("../../../dataset/overlappingnums/"+name_file+"/"+name_file+"_sub_feature_train" + enc + ".csv")

data = df.values
X_train_feature = data[:, 1:23].astype(int)
X_train_feature_ = data[:, 0:1].astype(int)
y_train_feature = data[:, -1].astype(int) - 1

df = read_csv("../../../dataset/overlappingnums/"+name_file+"/"+name_file+"_sub_feature_validate" + enc + ".csv")
data = df.values
X_validate_feature = data[:, 1:23].astype(int)
X_validate_feature_ = data[:, 0:1].astype(int)
y_validate_feature = data[:, -1].astype(int) - 1

df = read_csv("../../../dataset/overlappingnums/"+name_file+"/"+name_file+"_sub_feature_test" + enc + ".csv")
data = df.values
X_test_feature = data[:, 1:23].astype(int)
X_test_feature_ = data[:, 0:1].astype(int)
y_test_feature = data[:, -1].astype(int) - 1





X_feature = np.concatenate((X_train_feature, X_validate_feature, X_test_feature))
X_feature_id = np.concatenate((X_train_feature_, X_validate_feature_, X_test_feature_))
y_feature = np.concatenate((y_train_feature, y_validate_feature, y_test_feature))


# 共有6个模型，将数据均分为7份，其中6份用于K折验证，1份用于测试
share_size = int(X_feature_id.shape[0] / 7)

X_feature_train = X_feature[0:share_size * 5, :]
X_feature_val = X_feature[share_size * 5:share_size * 6, :]
X_feature_test = X_feature[share_size * 6:, :]
y_feature_train = y_feature[0:share_size * 5]
y_feature_train = y_feature[share_size * 5:share_size * 6, :]
y_feature_test = y_feature[share_size * 6:]

for i in range(2, 10):
    X_feature_train_tmp = np.concatenate((X_feature_train[:, 0:10].astype(int), X_feature_train[:, 10:10+i].astype(int), X_feature_train[:, 20:].astype(int)), axis=1)
    X_feature_test_tmp = np.concatenate((X_feature_test[:, 0:10].astype(int), X_feature_test[:, 10:10+i].astype(int), X_feature_test[:, 20:].astype(int)), axis=1)
    y_feature_train_tmp = np.array(y_feature_train)
    y_feature_test_tmp = np.array(y_feature_test)

    # 分裂属性的训练
    bst_feature = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=1, objective='multi:softprob')
    bst_feature.fit(X_feature_train_tmp, y_feature_train_tmp)
    acc_feature = accuracy_score(y_feature_test_tmp, bst_feature.predict(X_feature_test_tmp))
    print("acc feature:", acc_feature)



