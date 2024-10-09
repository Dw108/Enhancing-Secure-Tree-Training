# 训练集、测试集7：2：1
import copy
import itertools
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.stats import entropy, gaussian_kde
from numpy import transpose
import csv
from script.ope.Kerschbaum.Kerschbaum_encryption import Tree, encrypt_uni
from multiprocessing import Pool, cpu_count


def threadTask(data, i):
    tmp_col = []
    encrypt_tree = Tree()
    for j in range(len(data)):
        col = encrypt_uni(encrypt_tree, data[j][:, i].reshape((-1, 1)).T.tolist()[0], 0, 2000000)
        tmp_col.append(np.array([col]).T)
    return tmp_col


def write_data(address, name, data):
    with open(address, "w", encoding="utf-8", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(name)
        csv_writer.writerows(data)
        f.close()

# def no_overlap(name, arr_train, arr_validate, arr_test, scale1, scale2):
#     arr_train = arr_train.tolist()
#     for i in range(len(arr_train)):
#         arr_train[i][1] = float(arr_train[i][1]) + np.random.laplace(loc=0, scale=scale1)
#         arr_train[i][2] = float(arr_train[i][2]) + np.random.laplace(loc=0, scale=scale2)
#     arr_validate = arr_validate.tolist()
#     for i in range(len(arr_validate)):
#         arr_validate[i][1] = float(arr_validate[i][1]) + np.random.laplace(loc=0, scale=scale1)
#         arr_validate[i][2] = float(arr_validate[i][2]) + np.random.laplace(loc=0, scale=scale2)
#     arr_test = arr_test.tolist()
#     for i in range(len(arr_test)):
#         arr_test[i][1] = float(arr_test[i][1]) + np.random.laplace(loc=0, scale=scale1)
#         arr_test[i][2] = float(arr_test[i][2]) + np.random.laplace(loc=0, scale=scale2)
#     write_data("../../../dataset/bias/poker/poker_train" + str(scale1) + '_' + str(scale2) + ".csv",
#                name_feature, train_sub_feature_1)
def sub_feature(name, arr_train, arr_validate, arr_test, scale1, scale2):
    name_feature = copy.deepcopy(name)
    name_feature.insert(2, 'S1_sub')
    name_feature.insert(4, 'C1_sub')
    train_sub_feature_1 = []
    validate_sub_feature_1 = []
    test_sub_feature_1 = []
    arr_train = arr_train.astype(float)
    arr_validate = arr_validate.astype(float)
    arr_test = arr_test.astype(float)
    for i in range(len(arr_train)):
        arr_train[i] = arr_train[i] * 100
        arr_train[i][0] = arr_train[i][0] / 100
        arr_train[i][-1] = arr_train[i][-1] / 100
    for i in range(len(arr_validate)):
        arr_validate[i] = arr_validate[i] * 100
        arr_validate[i][0] = arr_validate[i][0] / 100
        arr_validate[i][-1] = arr_validate[i][-1] / 100
    for i in range(len(arr_test)):
        arr_test[i] = arr_test[i] * 100
        arr_test[i][0] = arr_test[i][0] / 100
        arr_test[i][-1] = arr_test[i][-1] / 100
    for i in range(len(arr_train)):
        tmp = arr_train[i].astype(float)
        tmp = np.insert(tmp, 2, float(tmp[1]) + np.random.laplace(loc=0, scale=scale1))
        tmp = np.insert(tmp, 4, float(tmp[3]) + np.random.laplace(loc=0, scale=scale2))
        train_sub_feature_1.append(tmp)

    for i in range(len(arr_validate)):
        tmp = arr_validate[i].astype(float)
        tmp = np.insert(tmp, 2, float(tmp[1]) + np.random.laplace(loc=0, scale=scale1))
        tmp = np.insert(tmp, 4, float(tmp[3]) + np.random.laplace(loc=0, scale=scale2))
        validate_sub_feature_1.append(tmp)

    for i in range(len(arr_test)):
        tmp = arr_test[i].astype(float)
        tmp = np.insert(tmp, 2, float(tmp[1]) + np.random.laplace(loc=0, scale=scale1))
        tmp = np.insert(tmp, 4, float(tmp[3]) + np.random.laplace(loc=0, scale=scale2))
        test_sub_feature_1.append(tmp)
    # 分裂属性
    # 未加密
    write_data("../../../dataset/bias/poker/poker_sub_feature_train" + str(scale1) + '_' + str(scale2) + ".csv",
               name_feature, train_sub_feature_1)

    write_data("../../../dataset/bias/poker/poker_sub_feature_validate" + str(scale1) + '_' + str(scale2) + ".csv",
               name_feature, validate_sub_feature_1)
    write_data("../../../dataset/bias/poker/poker_sub_feature_test" + str(scale1) + '_' + str(scale2) + ".csv",
               name_feature, test_sub_feature_1)
    # 加密
    uni_data_1_1 = cols_enc([train_sub_feature_1, validate_sub_feature_1, test_sub_feature_1])
    write_data("../../../dataset/bias/poker/poker_sub_feature_train_enc" + str(scale1) + '_' + str(scale2) + ".csv",
               name_feature, uni_data_1_1[0])
    write_data("../../../dataset/bias/poker/poker_sub_feature_validate_enc" + str(scale1) + '_' + str(scale2) + ".csv",
               name_feature, uni_data_1_1[1])
    write_data("../../../dataset/bias/poker/poker_sub_feature_test_enc" + str(scale1) + '_' + str(scale2) + ".csv",
               name_feature, uni_data_1_1[2])


def sub_item(name, arr_train, arr_validate, arr_test, scale1, scale2):
    arr_train = arr_train.tolist()
    for i in range(len(arr_train)):
        arr_train[i] = np.array(arr_train[i]).astype(float) * 100
        arr_train[i][0] = arr_train[i][0]/100
        arr_train[i][-1] = arr_train[i][-1] / 100
    arr_validate = arr_validate.tolist()
    for i in range(len(arr_validate)):
        arr_validate[i] = np.array(arr_validate[i]).astype(float) * 100
        arr_validate[i][0] = arr_validate[i][0] / 100
        arr_validate[i][-1] = arr_validate[i][-1] / 100
    arr_test = arr_test.tolist()
    for i in range(len(arr_test)):
        arr_test[i] = np.array(arr_test[i]).astype(float) * 100
        arr_test[i][0] = arr_test[i][0] / 100
        arr_test[i][-1] = arr_test[i][-1] / 100
    for i in range(len(arr_train)):
        tmp = copy.deepcopy(arr_train[i])
        tmp[1] = float(tmp[1]) + np.random.laplace(loc=0, scale=scale1)
        arr_train.append(tmp)
    for i in range(len(arr_validate)):
        tmp = copy.deepcopy(arr_validate[i])
        tmp[1] = float(tmp[1]) + np.random.laplace(loc=0, scale=scale1)
        arr_validate.append(tmp)


    for i in range(len(arr_test)):
        tmp = copy.deepcopy(arr_test[i])
        tmp[1] = float(tmp[1]) + np.random.laplace(loc=0, scale=scale1)
        arr_test.append(tmp)

    for i in range(len(arr_train)):
        tmp = copy.deepcopy(arr_train[i])
        tmp[2] = float(tmp[2]) + np.random.laplace(loc=0, scale=scale2)
        arr_train.append(tmp)

    for i in range(len(arr_validate)):
        tmp = copy.deepcopy(arr_validate[i])
        tmp[2] = float(tmp[2]) + np.random.laplace(loc=0, scale=scale2)
        arr_validate.append(tmp)

    for i in range(len(arr_test)):
        tmp = copy.deepcopy(arr_test[i])
        tmp[2] = float(tmp[2]) + np.random.laplace(loc=0, scale=scale2)
        arr_test.append(tmp)
    # arr_train = np.array(arr_train)
    # arr_validate = np.array(arr_validate)
    # arr_test = np.array(arr_test)
    # np.random.shuffle(arr_train)
    # split = int(arr_train.shape[0] * 0.25)
    # arr_train = arr_train[0:split]
    # np.random.shuffle(arr_validate)
    # split = int(arr_validate.shape[0] * 0.25)
    # arr_validate = arr_validate[0:split]
    # np.random.shuffle(arr_test)
    # split = int(arr_test.shape[0] * 0.25)
    # arr_test = arr_test[0:split]
    arr_train = np.array(arr_train)
    arr_validate = np.array(arr_validate)
    arr_test = np.array(arr_test)
    np.random.shuffle(arr_train)
    np.random.shuffle(arr_validate)
    np.random.shuffle(arr_test)
    arr_train_tmp = []
    set_train = set()
    for i in range(arr_train.shape[0]):
        if arr_train[i][0] not in set_train:
            arr_train_tmp.append(arr_train[i])
            set_train.add(arr_train[i][0])
    arr_validate_tmp = []
    set_validate = set()
    for i in range(arr_validate.shape[0]):
        if arr_validate[i][0] not in set_validate:
            arr_validate_tmp.append(arr_validate[i])
            set_validate.add(arr_validate[i][0])
    arr_test_tmp = []
    set_test = set()
    for i in range(arr_test.shape[0]):
        if arr_test[i][0] not in set_test:
            arr_test_tmp.append(arr_test[i])
            set_test.add(arr_test[i][0])
    # 分裂条目
    # 未加密
    write_data("../../../dataset/bias/poker/poker_sub_item_train" + str(scale1) + '_' + str(scale2) + ".csv", name, arr_train_tmp)
    write_data("../../../dataset/bias/poker/poker_sub_item_validate" + str(scale1) + '_' + str(scale2) + ".csv", name, arr_validate_tmp)
    write_data("../../../dataset/bias/poker/poker_sub_item_test" + str(scale1) + '_' + str(scale2) + ".csv", name, arr_test_tmp)

    # 加密
    uni_data_2 = cols_enc([arr_train_tmp, arr_validate_tmp, arr_test_tmp])
    write_data("../../../dataset/bias/poker/poker_sub_item_train_enc" + str(scale1) + '_' + str(scale2) + ".csv", name, uni_data_2[0])
    write_data("../../../dataset/bias/poker/poker_sub_item_validate_enc" + str(scale1) + '_' + str(scale2) + ".csv", name, uni_data_2[1])
    write_data("../../../dataset/bias/poker/poker_sub_item_test_enc" + str(scale1) + '_' + str(scale2) + ".csv", name, uni_data_2[2])
def cols_enc(data):
    # data全维度数据
    cols_train = []
    cols_val = []
    cols_test = []
    data_tmp = []
    for item in range(len(data)):
        item = np.array(data[item])
        item = item.astype(np.float64)
        item = item.astype(int)
        data_tmp.append(item)
    ids_train = (data_tmp[0][:, 0].reshape((-1, 1))).tolist()
    cols_train.append(ids_train)
    ids_val = (data_tmp[1][:, 0].reshape((-1, 1))).tolist()
    cols_val.append(ids_val)
    ids_test = (data_tmp[2][:, 0].reshape((-1, 1))).tolist()
    cols_test.append(ids_test)
    labels_train = (data_tmp[0][:, data_tmp[0].shape[1] - 1].reshape((-1, 1))).tolist()
    labels_val = (data_tmp[1][:, data_tmp[1].shape[1] - 1].reshape((-1, 1))).tolist()
    labels_test = (data_tmp[2][:, data_tmp[2].shape[1] - 1].reshape((-1, 1))).tolist()
    cols = [cols_train, cols_val, cols_test]
    labels = [labels_train, labels_val, labels_test]
    # 每一列的循环转换成多线程，多线程函数的输入是datatmp
    commnd = []
    for i in range(1, data_tmp[0].shape[1] - 1):
        commnd.append(i)
        # encrypt_tree = Tree()
        # for j in range(len(data_tmp)):
        #     # col = fhope.uni_batch(data_tmp[j][:, i].reshape((-1, 1)).tolist())
        #     col = encrypt_uni(encrypt_tree, data_tmp[j][:, i].reshape((-1, 1)).T.tolist()[0], 0, 500000)
        #     # 训练测试验证
        #     cols[j].append(np.array([col]).T)
    with ThreadPoolExecutor(max_workers=32) as executor:
        results = executor.map(threadTask, [data_tmp] * len(commnd), commnd)
        results_list = list(results)
    for i in range(0, data_tmp[0].shape[1] - 2):
        for j in range(len(data_tmp)):
            cols[j].append(results_list[i][j])
    for i in range(len(cols)):
        cols[i].append(labels[i])
    res = []
    for i in range(len(cols)):
        res.append(np.concatenate(cols[i], axis=1))
    return res


def main():
    csv_reader = csv.reader(open("../../../dataset/pokerhand/pokerhand.csv"))
    tmp = []
    for row in csv_reader:
        tmp.append(row)
    name = tmp[0]

    arr = tmp[1:]
    arr = arr[0:]
    arr = np.array(arr)
    np.random.shuffle(arr)
    arr = arr[0:]

    split_1 = int(len(arr) * 0.7)
    split_2 = int(len(arr) * 0.2)
    arr_train = arr[0:split_1]
    arr_validate = arr[split_1:split_1 + split_2]
    arr_test = arr[split_1 + split_2:]
    # 无重叠数据
    # 未加密
    write_data("../../../dataset/bias/poker/poker_train.csv", name, arr_train)
    write_data("../../../dataset/bias/poker/poker_validate.csv", name, arr_validate)
    write_data("../../../dataset/bias/poker/poker_test.csv", name, arr_test)
    # 无重叠 加密
    uni_data_0 = cols_enc([arr_train, arr_validate, arr_test])
    write_data("../../../dataset/bias/poker/poker_train_enc.csv", name, uni_data_0[0])
    write_data("../../../dataset/bias/poker/poker_validate_enc.csv", name, uni_data_0[1])
    write_data("../../../dataset/bias/poker/poker_test_enc.csv", name, uni_data_0[2])

    # 创建子属性的方式
    sub_feature(name, arr_train, arr_validate, arr_test, scale1=1000 / 1, scale2=1000 / 1)
    sub_feature(name, arr_train, arr_validate, arr_test, scale1=100 / 1, scale2=100 / 1)
    sub_feature(name, arr_train, arr_validate, arr_test, scale1=10 / 1, scale2=10 / 1)

    # 创造子项的方式
    sub_item(name, arr_train, arr_validate, arr_test, scale1=1000 / 1, scale2=1000 / 1)
    sub_item(name, arr_train, arr_validate, arr_test, scale1=100 / 1, scale2=100 / 1)
    sub_item(name, arr_train, arr_validate, arr_test, scale1=10 / 1, scale2=10 / 1)


if __name__ == '__main__':  # 不加这句就会报错
    main()
