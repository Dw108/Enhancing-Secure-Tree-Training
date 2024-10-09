import csv

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

for i in range(2, 10):
    df = read_csv("../../../dataset/overlappingnums/covtype/mid/data_feature_train_" + str(i) + ".csv")
    feature_train1 = df.values.astype(float)
    df = read_csv("../../../dataset/overlappingnums/covtype/mid/data_feature_val_" + str(i) + ".csv")
    feature_train2 = df.values.astype(float)
    feature_train = np.concatenate((feature_train1, feature_train2), axis=0)
    df = read_csv("../../../dataset/overlappingnums/covtype/mid/data_feature_test_" + str(i) + ".csv")
    feature_test = df.values.astype(float)

    df = read_csv("../../../dataset/overlappingnums/covtype/mid/data_select_train_" + str(i) + ".csv")
    select_train1 = df.values.astype(float)
    df = read_csv("../../../dataset/overlappingnums/covtype/mid/data_select_val_" + str(i) + ".csv")
    select_train2 = df.values.astype(float)
    select_train = np.concatenate((select_train1, select_train2), axis=0)
    df = read_csv("../../../dataset/overlappingnums/covtype/mid/data_select_test_" + str(i) + ".csv")
    select_test = df.values.astype(float)

    df = read_csv("../../../dataset/overlappingnums/covtype/mid/data_item_train_" + str(i) + ".csv")
    item_train1 = df.values.astype(float)
    df = read_csv("../../../dataset/overlappingnums/covtype/mid/data_item_val_" + str(i) + ".csv")
    item_train2 = df.values.astype(float)
    item_train = np.concatenate((item_train1, item_train2), axis=0)
    df = read_csv("../../../dataset/overlappingnums/covtype/mid/data_item_test_" + str(i) + ".csv")
    item_test = df.values.astype(float)



    data_train = np.concatenate((feature_train, select_train, item_train), axis=1)
    data_test = np.concatenate((feature_test, select_test, item_test), axis=1)

    df = read_csv("../../../dataset/overlappingnums/covtype/mid/label_feature_train.csv")
    label_train1 = df.values.astype(int).T[0]

    df = read_csv("../../../dataset/overlappingnums/covtype/mid/label_feature_val.csv")
    label_train2 = df.values.astype(int).T[0]
    label_train = np.concatenate((label_train1,label_train2))
    df = read_csv("../../../dataset/overlappingnums/covtype/mid/label_feature_test.csv")
    label_test = df.values.astype(int).T[0]

    b_size = 256
    max_epochs = 30
    num_iteration = 1
    for ep in range(max_epochs-1, max_epochs):
        acc_max = 0
        for items in range(num_iteration):
            # 2. 定义模型
            init = K.initializers.glorot_uniform(seed=None)
            simple_adam = K.optimizers.Adam()
            model = K.models.Sequential()
            # units 是输出大小
            model.add(K.layers.Dense(units=35, input_dim=21, activation='relu'))
            # model.add(K.layers.Dense(units=10, kernel_initializer=init, activation='relu'))
            model.add(K.layers.Dense(units=21, activation='relu'))
            model.add(K.layers.Dense(units=7, activation='softmax'))
            model.compile(loss='sparse_categorical_crossentropy', optimizer=simple_adam, metrics=['accuracy'])
            print("iteration:", items)
            print("Starting training ")
            h = model.fit(data_train, label_train, batch_size=b_size, epochs=ep + 1, shuffle=True, verbose=1)
            print("Training finished \n")
            pred = numpy.argmax(model.predict(data_test), axis=1)
            acc = accuracy_score(label_test, pred)
            print("iteration:", i, "pre acc:", acc)
        #     if acc > acc_max:
        #         acc_max = acc
        # print("acc_max: ", acc_max)
