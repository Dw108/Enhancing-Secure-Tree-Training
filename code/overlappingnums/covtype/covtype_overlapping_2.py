# 将两种训练方式得到的模型在验证集上得到的预测结果进行融合
# 融合涉及id对齐
# 元学习器选择一个简单的神经网络
import csv
import copy

from matplotlib import pyplot
from sklearn.feature_selection import SelectFromModel
from numpy import sort
from xgboost import XGBClassifier, plot_importance
import numpy as np
from pandas import read_csv
from sklearn.metrics import accuracy_score


# enc = ''
enc = '_enc'
name_file = 'covtype'
dp_scale = 0.0001
def write_data(address, name, data):
    with open(address, "w", encoding="utf-8", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(name)
        csv_writer.writerows(data)
        f.close()

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

def dp_noise(data_tmp, scale_tmp):
    mean_value = np.average(data_tmp, axis=0)
    for i in range(data_tmp.shape[1]):
        true_scale = mean_value[i] * scale_tmp
        for j in range(data_tmp.shape[0]):
            data_tmp[j][i] = data_tmp[j][i] + np.random.laplace(loc=0, scale=true_scale)
    return data_tmp
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
y_feature_val = y_feature[share_size * 5:share_size * 6]
y_feature_test = y_feature[share_size * 6:]
# write_data("../../../dataset/overlappingnums/covtype/mid/label_feature_val.csv", ['label'],
#            np.array([y_feature_val]).T)
# write_data("../../../dataset/overlappingnums/covtype/mid/label_feature_test.csv", ['label'],
#            np.array([y_feature_test]).T)
for i in range(9, 10):
    X_feature_train_tmp = np.concatenate((X_feature_train[:, 0:10].astype(int), X_feature_train[:, 10:10+i].astype(int), X_feature_train[:, 20:].astype(int)), axis=1)
    X_feature_train_tmp = dp_noise(X_feature_train_tmp, dp_scale)
    X_feature_val_tmp = np.concatenate((X_feature_val[:, 0:10].astype(int), X_feature_val[:, 10:10+i].astype(int), X_feature_val[:, 20:].astype(int)), axis=1)
    X_feature_val_tmp = dp_noise(X_feature_val_tmp, dp_scale)
    X_feature_test_tmp = np.concatenate((X_feature_test[:, 0:10].astype(int), X_feature_test[:, 10:10 + i].astype(int),
                                        X_feature_test[:, 20:].astype(int)), axis=1)
    X_feature_test_tmp = dp_noise(X_feature_test_tmp, dp_scale)
    y_feature_train_tmp = np.array(y_feature_train)
    y_feature_val_tmp = np.array(y_feature_val)
    y_feature_test_tmp = np.array(y_feature_test)



    model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=1, objective='multi:softprob')
    model.fit(X_feature_train_tmp, y_feature_train_tmp)
    print(model.feature_importances_)
    plot_importance(model)
    pyplot.show()
    thresholds = sort(model.feature_importances_)
    max_acc = 0
    best_model = 0
    best_X_train = 0
    best_X_val = 0
    best_X_test = 0
    for thresh in thresholds:
        # select features using threshold
        selection = SelectFromModel(model, threshold=thresh, prefit=True)
        select_X_train = selection.transform(X_feature_train_tmp)
        # train model
        selection_model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=1, objective='multi:softprob')
        selection_model.fit(select_X_train, y_feature_train_tmp)
        # eval model
        select_X_val = selection.transform(X_feature_val_tmp)
        y_pred = selection_model.predict(select_X_val)
        predictions = [round(value) for value in y_pred]
        accuracy = accuracy_score(y_feature_val_tmp, predictions)
        if accuracy > max_acc:
            max_acc = accuracy
            best_model = copy.deepcopy(selection_model)
            best_X_train = copy.deepcopy(selection.transform(X_feature_train_tmp))
            best_X_val = copy.deepcopy(selection.transform(X_feature_val_tmp))
            best_X_test = copy.deepcopy(selection.transform(X_feature_test_tmp))
        print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy * 100.0))



    # 分裂属性的训练
    # bst_feature = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=1, objective='multi:softprob')
    acc_feature = accuracy_score(y_feature_val_tmp, best_model.predict(best_X_val))
    acc_feature_test = accuracy_score(y_feature_test_tmp, best_model.predict(best_X_test))
    accumulation_train = best_model.predict(best_X_train, output_margin=True)
    accumulation_val = best_model.predict(best_X_val, output_margin=True)
    accumulation_test = best_model.predict(best_X_test, output_margin=True)
    # name_feature =['1', '2', '3', '4', '5', '6', '7']
    # write_data("../../../dataset/overlappingnums/covtype/mid/data_select_train_" + str(i) + ".csv", name_feature,
    #            accumulation_train)
    # write_data("../../../dataset/overlappingnums/covtype/mid/data_select_val_"+str(i)+".csv", name_feature, accumulation_val)
    # write_data("../../../dataset/overlappingnums/covtype/mid/data_select_test_" + str(i) + ".csv", name_feature, accumulation_test)

    print("acc select:", acc_feature)
    print("acc select test:", acc_feature_test)



