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
from xgboost import XGBClassifier
import numpy as np
from pandas import read_csv
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import script.network
import torch.utils.data as Data
import hiddenlayer as hl

from script.bias_exp.CASP.CASP_train_test_split_enc_multiThread import write_data

# enc = ''
enc = '_enc'
name_file = 'covtype'
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



for i in range(2, 10):
    df = read_csv("../../../dataset/overlappingnums/" + name_file + "/" + name_file + "_sub_item_train_" + str(i)+'_enc' + ".csv")

    data = df.values
    X_train_feature = data[:, 1:13].astype(int)
    X_train_feature = dp_noise(X_train_feature, dp_scale)
    X_train_feature_ = data[:, 0:1].astype(int)
    y_train_feature = data[:, -1].astype(int) - 1

    df = read_csv("../../../dataset/overlappingnums/" + name_file + "/" + name_file + "_sub_item_validate_" + str(i)+'enc' + ".csv")
    data = df.values
    X_validate_feature = data[:, 1:13].astype(int)
    X_validate_feature = dp_noise(X_validate_feature, dp_scale)
    X_validate_feature_ = data[:, 0:1].astype(int)
    y_validate_feature = data[:, -1].astype(int) - 1

    df = read_csv("../../../dataset/overlappingnums/" + name_file + "/" + name_file + "_sub_item_test_" + str(i)+'enc' + ".csv")
    data = df.values
    X_test_feature = data[:, 1:13].astype(int)
    X_test_feature = dp_noise(X_test_feature, dp_scale)
    X_test_feature_ = data[:, 0:1].astype(int)
    y_test_feature = data[:, -1].astype(int) - 1

    X_feature = np.concatenate((X_train_feature, X_validate_feature, X_test_feature))
    X_feature_id = np.concatenate((X_train_feature_, X_validate_feature_, X_test_feature_))
    y_feature = np.concatenate((y_train_feature, y_validate_feature, y_test_feature))

    # 共有6个模型，将数据均分为7份，其中6份用于K折验证，1份用于测试
    share_size = int(X_feature_id.shape[0] / 7)

    # X_feature_train = X_feature[0:share_size * 6, :]
    # X_feature_test = X_feature[share_size * 6:, :]
    # y_feature_train = y_feature[0:share_size * 6]
    # y_feature_test = y_feature[share_size * 6:]

    X_feature_train = X_feature[0:share_size * 5, :]
    X_feature_val = X_feature[share_size * 5:share_size * 6, :]
    X_feature_test = X_feature[share_size * 6:, :]
    y_feature_train = y_feature[0:share_size * 5]
    y_feature_val = y_feature[share_size * 5:share_size * 6]
    y_feature_test = y_feature[share_size * 6:]


    # 分裂属性的训练
    time_start = time.time()
    bst_feature = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=1, objective='multi:softprob')
    bst_feature.fit(X_feature_train, y_feature_train)
    acc_feature = accuracy_score(y_feature_val, bst_feature.predict(X_feature_val))
    acc_feature_test = accuracy_score(y_feature_test, bst_feature.predict(X_feature_test))

    accumulation_train = bst_feature.predict(X_feature_train, output_margin=True)
    accumulation_val = bst_feature.predict(X_feature_val, output_margin=True)
    accumulation_test = bst_feature.predict(X_feature_test, output_margin=True)
    name_feature = ['1', '2', '3', '4', '5', '6', '7']
    write_data("../../../dataset/overlappingnums/covtype/mid/data_item_train_" + str(i) + ".csv", name_feature,
               accumulation_train)
    write_data("../../../dataset/overlappingnums/covtype/mid/data_item_val_" + str(i) + ".csv", name_feature,
               accumulation_val)
    write_data("../../../dataset/overlappingnums/covtype/mid/data_item_test_" + str(i) + ".csv", name_feature,
               accumulation_test)

    print("acc feature:", acc_feature)
    print("acc feature test:", acc_feature_test)


