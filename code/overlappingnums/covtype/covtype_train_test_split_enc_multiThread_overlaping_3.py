# 训练集、测试集7：2：1
import copy
import itertools
import random
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


def sub_item(name, arr_train, arr_validate, arr_test):
    for max_overlap in range(2, 10):
        print(max_overlap)
        arr_train_base = np.concatenate((arr_train[:, 0:11].astype(int), arr_train[:, 21:].astype(int)), axis=1)
        arr_validate_base = np.concatenate((arr_validate[:, 0:11].astype(int), arr_validate[:, 21:].astype(int)),
                                           axis=1)
        arr_test_base = np.concatenate((arr_test[:, 0:11].astype(int), arr_test[:, 21:].astype(int)), axis=1)
        for i in range(1, max_overlap + 1):
            for item in range(len(arr_train_base)):
                condition = arr_train[:, 0] == arr_train_base[item][0]
                filtered_data = arr_train[condition][0]
                if random.random() > 0.5:
                    arr_train_base[item][i] = filtered_data[i + 10]
        for i in range(1, max_overlap + 1):
            for item in range(len(arr_validate_base)):
                condition = arr_validate[:, 0] == arr_validate_base[item][0]
                filtered_data = arr_validate[condition][0]
                if random.random() > 0.5:
                    arr_validate_base[item][i] = filtered_data[i + 10]
        for i in range(1, max_overlap + 1):
            for item in range(len(arr_test_base)):
                condition = arr_test[:, 0] == arr_test_base[item][0]
                filtered_data = arr_test[condition][0]
                if random.random() > 0.5:
                    arr_test_base[item][i] = filtered_data[i + 10]
        write_data("../../../dataset/overlappingnums/covtype/covtype_sub_item_train_" + str(max_overlap) + ".csv", name,
                   arr_train_base)
        write_data("../../../dataset/overlappingnums/covtype/covtype_sub_item_validate_" + str(max_overlap) + ".csv",
                   name, arr_validate_base)
        write_data("../../../dataset/overlappingnums/covtype/covtype_sub_item_test_" + str(max_overlap) + ".csv", name,
                   arr_test_base)
        # # 去重
        # arr_train = np.array(arr_train)
        # arr_validate = np.array(arr_validate)
        # arr_test = np.array(arr_test)
        # np.random.shuffle(arr_train)
        # np.random.shuffle(arr_validate)
        # np.random.shuffle(arr_test)
        # arr_train_tmp = []
        # set_train = set()
        # for i in range(arr_train.shape[0]):
        #     if arr_train[i][0] not in set_train:
        #         arr_train_tmp.append(arr_train[i])
        #         set_train.add(arr_train[i][0])
        # arr_validate_tmp = []
        # set_validate = set()
        # for i in range(arr_validate.shape[0]):
        #     if arr_validate[i][0] not in set_validate:
        #         arr_validate_tmp.append(arr_validate[i])
        #         set_validate.add(arr_validate[i][0])
        # arr_test_tmp = []
        # set_test = set()
        # for i in range(arr_test.shape[0]):
        #     if arr_test[i][0] not in set_test:
        #         arr_test_tmp.append(arr_test[i])
        #         set_test.add(arr_test[i][0])
        # # 分裂条目
        # # 未加密
        # write_data("../../../dataset/overlappingnums/covtype/covtype_sub_item_train.csv", name, arr_train_tmp)
        # write_data("../../../dataset/overlappingnums/covtype/covtype_sub_item_validate.csv", name, arr_validate_tmp)
        # write_data("../../../dataset/overlappingnums/covtype/covtype_sub_item_test.csv", name, arr_test_tmp)

    # # 加密
    # uni_data_2 = cols_enc([arr_train_tmp, arr_validate_tmp, arr_test_tmp])
    # write_data("../../../dataset/overlappingnums/covtype/covtype_sub_item_train_enc.csv", name, uni_data_2[0])
    # write_data("../../../dataset/overlappingnums/covtype/covtype_sub_item_validate_enc.csv", name, uni_data_2[1])
    # write_data("../../../dataset/overlappingnums/covtype/covtype_sub_item_test_enc.csv", name, uni_data_2[2])


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
    ids_train = data_tmp[0][:, 0].reshape((-1, 1)).tolist()
    cols_train.append(ids_train)
    ids_val = data_tmp[1][:, 0].reshape((-1, 1)).tolist()
    cols_val.append(ids_val)
    ids_test = data_tmp[2][:, 0].reshape((-1, 1)).tolist()
    cols_test.append(ids_test)
    labels_train = data_tmp[0][:, data_tmp[0].shape[1] - 1].reshape((-1, 1)).tolist()
    labels_val = data_tmp[1][:, data_tmp[1].shape[1] - 1].reshape((-1, 1)).tolist()
    labels_test = data_tmp[2][:, data_tmp[2].shape[1] - 1].reshape((-1, 1)).tolist()
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
    csv_reader_train = csv.reader(open("../../../dataset/overlappingnums/covtype/covtype_sub_feature_train.csv"))
    csv_reader_val = csv.reader(open("../../../dataset/overlappingnums/covtype/covtype_sub_feature_validate.csv"))
    csv_reader_test = csv.reader(open("../../../dataset/overlappingnums/covtype/covtype_sub_feature_test.csv"))
    tmp_train = []
    tmp_val = []
    tmp_test = []
    for row in csv_reader_train:
        tmp_train.append(row)
    name = ["id", "Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
            "Horizontal_Distance_To_Roadways", "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
            "Horizontal_Distance_To_Fire_Points", "Wilderness_Area", "Soil_Type", "label"]

    for row in csv_reader_val:
        tmp_val.append(row)
    for row in csv_reader_test:
        tmp_test.append(row)

    arr_train = tmp_train[1:]
    arr_train = np.array(arr_train).astype(float).astype(int)

    arr_val = tmp_val[1:]
    arr_val = np.array(arr_val).astype(float).astype(int)

    arr_test = tmp_test[1:]
    arr_test = np.array(arr_test).astype(float).astype(int)

    # 无重叠数据
    # # 未加密
    # write_data("../../../dataset/bias/covtype/covtype_train.csv", name, arr_train)
    # write_data("../../../dataset/bias/covtype/covtype_validate.csv", name, arr_validate)
    # write_data("../../../dataset/bias/covtype/covtype_validate.csv", name, arr_test)
    # # 无重叠 加密
    # uni_data_0 = cols_enc([arr_train, arr_validate, arr_test])
    # write_data("../../../dataset/bias/covtype/covtype_train_enc.csv", name, uni_data_0[0])
    # write_data("../../../dataset/bias/covtype/covtype_validate_enc.csv", name, uni_data_0[1])
    # write_data("../../../dataset/bias/covtype/covtype_test_enc.csv", name, uni_data_0[2])

    # 创建子属性的方式
    # sub_feature(name, arr_train, arr_validate, arr_test, scale1=100 / 1, scale2=10 / 1)
    # # sub_feature(name, arr_train, arr_validate, arr_test, scale1=10 / 1, scale2=1 / 1)
    # sub_feature(name, arr_train, arr_validate, arr_test, scale1=1000 / 1, scale2=100 / 1, scale3=10 / 1, scale4=100 / 1,
    #             scale5=50 / 1, scale6=1000 / 1, scale7=100 / 1, scale8=100 / 1, scale9=100 / 1, scale10=1000 / 1)

    # 创造子项的方式
    # sub_item(name, arr_train, arr_validate, arr_test, scale1=100 / 1, scale2=10 / 1)
    # sub_item(name, arr_train, arr_validate, arr_test, scale1=10 / 1, scale2=1 / 1)
    sub_item(name, arr_train, arr_val, arr_test)


if __name__ == '__main__':  # 不加这句就会报错
    main()
