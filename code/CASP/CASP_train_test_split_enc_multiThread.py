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

def threadTask(data,i):
    tmp_col = []
    encrypt_tree = Tree()
    for j in range(len(data)):
        col = encrypt_uni(encrypt_tree, data[j][:, i].reshape((-1, 1)).T.tolist()[0], 0, 2000000)
        tmp_col.append(np.array([col]).T)
    return tmp_col

def cols_enc(data):
    # data全维度数据
    cols_train = []
    cols_val = []
    cols_test = []
    data_tmp = []
    for item in range(len(data)):
        item = np.array(data[item])
        item = item.astype(np.float64)
        # item = item.astype(int)
        data_tmp.append(item)
    ids_train = data_tmp[0][:, 0].reshape((-1, 1)).tolist()
    cols_train.append(ids_train)
    ids_val = data_tmp[1][:, 0].reshape((-1, 1)).tolist()
    cols_val.append(ids_val)
    ids_test = data_tmp[2][:, 0].reshape((-1, 1)).tolist()
    cols_test.append(ids_test)
    labels_train = data_tmp[0][:, data_tmp[0].shape[1]-1].reshape((-1, 1)).tolist()
    labels_val = data_tmp[1][:, data_tmp[1].shape[1] - 1].reshape((-1, 1)).tolist()
    labels_test = data_tmp[2][:, data_tmp[2].shape[1] - 1].reshape((-1, 1)).tolist()
    cols = [cols_train, cols_val, cols_test]
    labels = [labels_train, labels_val, labels_test]
    # 每一列的循环转换成多线程，多线程函数的输入是datatmp
    commnd = []
    for i in range(1, data_tmp[0].shape[1]-1):
        commnd.append(i)
        # encrypt_tree = Tree()
        # for j in range(len(data_tmp)):
        #     # col = fhope.uni_batch(data_tmp[j][:, i].reshape((-1, 1)).tolist())
        #     col = encrypt_uni(encrypt_tree, data_tmp[j][:, i].reshape((-1, 1)).T.tolist()[0], 0, 500000)
        #     # 训练测试验证
        #     cols[j].append(np.array([col]).T)
    with ThreadPoolExecutor(max_workers=32) as executor:
        results = executor.map(threadTask, [data_tmp]*len(commnd), commnd)
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
    csv_reader = csv.reader(open("../../dataset/CASP/CASP_processed.csv"))
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


    uni_data_0 = cols_enc([arr_train, arr_validate, arr_test])
    # 无重叠数据
    with open("../../dataset/CASP/CASP_train_enc.csv", "w", encoding="utf-8", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(name)
        csv_writer.writerows(uni_data_0[0])
        f.close()

    with open("../../dataset/CASP/CASP_validate_enc.csv", "w", encoding="utf-8", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(name)
        csv_writer.writerows(uni_data_0[1])
        f.close()
    with open("../../dataset/CASP/CASP_test_enc.csv", "w", encoding="utf-8", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(name)
        csv_writer.writerows(uni_data_0[2])
        f.close()
    # 创建子属性的方式
    name_feature = copy.deepcopy(name)
    name_feature.insert(2, 'F1_sub')
    name_feature.insert(4, 'F2_sub')
    train_sub_feature = []
    validate_sub_feature = []
    test_sub_feature = []
    for i in range(len(arr_train)):
        tmp = arr_train[i].astype(float)
        tmp = np.insert(tmp, 2, float(tmp[1]) * 10)
        tmp = np.insert(tmp, 4, float(tmp[3]) * 10)
        train_sub_feature.append(tmp)

    for i in range(len(arr_validate)):
        tmp = arr_validate[i].astype(float)
        tmp = np.insert(tmp, 2, float(tmp[1]) * 10)
        tmp = np.insert(tmp, 4, float(tmp[3]) * 10)
        validate_sub_feature.append(tmp)

    for i in range(len(arr_test)):
        tmp = arr_test[i].astype(float)
        tmp = np.insert(tmp, 2, float(tmp[1]) * 10)
        tmp = np.insert(tmp, 4, float(tmp[3]) * 10)
        test_sub_feature.append(tmp)

    uni_data_1 = cols_enc([train_sub_feature, validate_sub_feature, test_sub_feature])

    with open("../../dataset/CASP/CASP_sub_feature_train_enc.csv", "w", encoding="utf-8", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(name_feature)
        csv_writer.writerows(uni_data_1[0])
        f.close()
    with open("../../dataset/CASP/CASP_sub_feature_validate_enc.csv", "w", encoding="utf-8", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(name_feature)
        csv_writer.writerows(uni_data_1[1])
        f.close()
    with open("../../dataset/CASP/CASP_sub_feature_test_enc.csv", "w", encoding="utf-8", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(name_feature)
        csv_writer.writerows(uni_data_1[2])
        f.close()


    # 创造子项的方式
    arr_train = arr_train.tolist()
    for i in range(len(arr_train)):
        tmp = copy.deepcopy(arr_train[i])
        tmp[1] = float(tmp[1])*10
        arr_train.append(tmp)

    arr_validate = arr_validate.tolist()
    for i in range(len(arr_validate)):
        tmp = copy.deepcopy(arr_validate[i])
        tmp[1] = float(tmp[1])*10
        arr_validate.append(tmp)

    arr_test = arr_test.tolist()
    for i in range(len(arr_test)):
        tmp = copy.deepcopy(arr_test[i])
        tmp[1] = float(tmp[1])*10
        arr_test.append(tmp)

    for i in range(len(arr_train)):
        tmp = copy.deepcopy(arr_train[i])
        tmp[2] = float(tmp[2])*10
        arr_train.append(tmp)

    for i in range(len(arr_validate)):
        tmp = copy.deepcopy(arr_validate[i])
        tmp[2] = float(tmp[2])*10
        arr_validate.append(tmp)

    for i in range(len(arr_test)):
        tmp = copy.deepcopy(arr_test[i])
        tmp[2] = float(tmp[2])*10
        arr_test.append(tmp)

    uni_data_2 = cols_enc([arr_train, arr_validate, arr_test])

    with open("../../dataset/CASP/CASP_sub_item_train_enc.csv", "w", encoding="utf-8", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(name)
        csv_writer.writerows(uni_data_2[0])
        f.close()
    with open("../../dataset/CASP/CASP_sub_item_validate_enc.csv", "w", encoding="utf-8", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(name)
        csv_writer.writerows(uni_data_2[1])
        f.close()

    with open("../../dataset/CASP/CASP_sub_item_test_enc.csv", "w", encoding="utf-8", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(name)
        csv_writer.writerows(uni_data_2[2])
        f.close()


if __name__=='__main__': #不加这句就会报错
    main()

