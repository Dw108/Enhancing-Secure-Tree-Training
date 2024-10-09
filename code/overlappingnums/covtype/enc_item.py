from pandas import read_csv
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import csv
from script.ope.Kerschbaum.Kerschbaum_encryption import Tree, encrypt_uni

enc = '_enc'
name_file = 'covtype'

def threadTask(data, i):
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


def write_data(address, name, data):
    with open(address, "w", encoding="utf-8", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(name)
        csv_writer.writerows(data)
        f.close()


for i in range(2, 10):
    df = read_csv("../../../dataset/overlappingnums/" + name_file + "/" + name_file + "_sub_item_train_" + str(i) + ".csv")
    arr_train_tmp = df.values

    df = read_csv("../../../dataset/overlappingnums/" + name_file + "/" + name_file + "_sub_item_validate_" + str(i) + ".csv")
    arr_validate_tmp = df.values

    df = read_csv("../../../dataset/overlappingnums/" + name_file + "/" + name_file + "_sub_item_test_" + str(i) + ".csv")
    arr_test_tmp = df.values
    name = ["id", "Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
            "Horizontal_Distance_To_Roadways", "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
            "Horizontal_Distance_To_Fire_Points", "Wilderness_Area", "Soil_Type", "label"]
    uni_data_2 = cols_enc([arr_train_tmp, arr_validate_tmp, arr_test_tmp])
    write_data("../../../dataset/overlappingnums/covtype/covtype_sub_item_train_" + str(i) + "_enc.csv", name, uni_data_2[0])
    write_data("../../../dataset/overlappingnums/covtype/covtype_sub_item_validate_" + str(i) + "enc.csv", name, uni_data_2[1])
    write_data("../../../dataset/overlappingnums/covtype/covtype_sub_item_test_" + str(i) + "enc.csv", name, uni_data_2[2])
