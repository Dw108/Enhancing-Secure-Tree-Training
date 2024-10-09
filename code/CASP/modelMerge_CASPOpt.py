# 将两种训练方式得到的模型在验证集上得到的预测结果进行融合
# 融合涉及id对齐
# 元学习器选择一个简单的神经网络
import time
import copy
from sklearn.feature_selection import SelectFromModel
from numpy import sort
import keras as K
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBClassifier, XGBRegressor
import numpy as np
from pandas import read_csv
from sklearn.metrics import accuracy_score, balanced_accuracy_score

# scale = '1000.0_100.0'
# scale = '100.0_10.0'
scale = '10.0_1.0'
# enc = ''
enc = '_enc'
name_file = 'CASP'
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
df = read_csv("../../dataset/bias/"+name_file+"/"+name_file+"_sub_feature_train"+enc+scale+".csv")
data = df.values
X_train_feature = data[:, 1:12].astype(int)
X_train_feature = dp_noise(X_train_feature, dp_scale)
X_train_feature_ = data[:, 0:1].astype(int)
y_train_feature = data[:, -1].astype(int)

df = read_csv("../../dataset/bias/"+name_file+"/"+name_file+"_sub_feature_validate"+enc+scale+".csv")
data = df.values
X_validate_feature = data[:, 1:12].astype(int)
X_validate_feature = dp_noise(X_validate_feature, dp_scale)
X_validate_feature_ = data[:, 0:1].astype(int)
y_validate_feature = data[:, -1].astype(int)


df = read_csv("../../dataset/bias/"+name_file+"/"+name_file+"_sub_feature_test"+enc+scale+".csv")
data = df.values
X_test_feature = data[:, 1:12].astype(int)
X_test_feature = dp_noise(X_test_feature, dp_scale)
X_test_feature_ = data[:, 0:1].astype(int)
y_test_feature = data[:, -1].astype(int)

# 分裂条目方式的数据
df = read_csv("../../dataset/bias/"+name_file+"/"+name_file+"_sub_item_train"+enc+scale+".csv")
data = df.values
X_train_item = data[:, 1:10]
X_train_item = dp_noise(X_train_item, dp_scale)
X_train_item_ = data[:, 0:1]
y_train_item = data[:, -1]


df = read_csv("../../dataset/bias/"+name_file+"/"+name_file+"_sub_item_validate"+enc+scale+".csv")
data = df.values
X_validate_item = data[:, 1:10]
X_validate_item = dp_noise(X_validate_item, dp_scale)
X_validate_item_ = data[:, 0:1]
y_validate_item = data[:, -1]

df = read_csv("../../dataset/bias/"+name_file+"/"+name_file+"_sub_item_test"+enc+scale+".csv")
data = df.values
X_test_item = data[:, 1:10]
X_test_item = dp_noise(X_test_item, dp_scale)
X_test_item_ = data[:, 0:1]
y_test_item = data[:, -1]

n_estimators = 15
max_depth = 4
# 分裂属性的训练
bst_feature = XGBRegressor(nthread=1, n_estimators=n_estimators, max_depth=max_depth, learning_rate=1, objective='reg:squarederror')
bst_feature.fit(X_train_feature, y_train_feature)
mse_feature1 = mean_squared_error(y_train_feature, bst_feature.predict(X_train_feature))
mse_feature = mean_squared_error(y_validate_feature, bst_feature.predict(X_validate_feature))
mse_feature_test = mean_squared_error(y_test_feature, bst_feature.predict(X_test_feature))
r2_feature = r2_score(y_validate_feature, bst_feature.predict(X_validate_feature))
r2_feature_test = r2_score(y_test_feature, bst_feature.predict(X_test_feature))
accumulation_feature = bst_feature.predict(X_validate_feature, output_margin=True)
accumulation_feature_test = bst_feature.predict(X_test_feature, output_margin=True)
print("mse feature1:", mse_feature1)
print("mse feature:", mse_feature)
print("mse feature test:", mse_feature_test)
print("r2 feature:", r2_feature)
print("r2 feature test:", r2_feature_test)


# 分裂条目的训练
bst_item = XGBRegressor(nthread=1,n_estimators=n_estimators, max_depth=max_depth, learning_rate=1, objective='reg:squarederror')
bst_item.fit(X_train_item, y_train_item)
# make predictions
mse_item = mean_squared_error(y_validate_item, bst_item.predict(X_validate_item))
mse_item_test = mean_squared_error(y_test_item, bst_item.predict(X_test_item))
r2_item = r2_score(y_validate_item, bst_item.predict(X_validate_item))
r2_item_test = r2_score(y_test_item, bst_item.predict(X_test_item))
accumulation_item = bst_item.predict(X_validate_item, output_margin=True)
accumulation_item_test = bst_item.predict(X_test_item, output_margin=True)
print("mse item:", mse_item)
print("mse item test:", mse_item_test)
print("r2 item:", r2_item)
print("r2 item test:", r2_item_test)


model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=1, objective='reg:squarederror')
model.fit(X_train_feature, y_train_feature)
y_pred = model.predict(X_validate_feature)
predictions = [round(value) for value in y_pred]
mse = mean_squared_error(y_validate_feature, predictions)
r2_score_tmp = r2_score(y_validate_feature, predictions)
print("mse :", mse)
print("r2_score :", r2_score_tmp)
thresholds = sort(model.feature_importances_)
min_mse = 1000000
best_model = 0
best_X_val = 0
best_X_test = 0
for thresh in thresholds:
    # select features using threshold
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    select_X_train = selection.transform(X_train_feature)
    # train model
    selection_model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=1, objective='reg:squarederror')
    selection_model.fit(select_X_train, y_train_feature)
    # eval model
    select_X_val = selection.transform(X_validate_feature)
    y_pred = selection_model.predict(select_X_val)
    predictions = [round(value) for value in y_pred]
    mse_temp = mean_squared_error(y_validate_feature, predictions)
    if mse_temp < min_mse:
        min_mse = mse_temp
        best_model = copy.deepcopy(selection_model)
        # best_X_train = copy.deepcopy(selection.transform(X_select_second_train))
        best_X_val = copy.deepcopy(selection.transform(X_validate_feature))
        best_X_test = copy.deepcopy(selection.transform(X_test_feature))
    print("Thresh=%.3f, n=%d, mse: %.2f" % (thresh, select_X_train.shape[1], min_mse))
y_pred = best_model.predict(best_X_val)
mse_select = mean_squared_error(y_validate_feature, best_model.predict(best_X_val))
mse_select_test = mean_squared_error(y_test_feature, best_model.predict(best_X_test))
r2_select = r2_score(y_validate_feature, best_model.predict(best_X_val))
r2_select_test = r2_score(y_test_feature, best_model.predict(best_X_test))
print("mse select:", mse_select)
print("mse select test:", mse_select_test)
print("r2 select:", r2_select)
print("r2 select test:", r2_select_test)


accumulation_select_second_train = best_model.predict(best_X_val, output_margin=True)
accumulation_select_second_test = best_model.predict(best_X_test, output_margin=True)



# 将其他数据集顺序和feature顺序同步
X_feature_train = accumulation_feature
y_feature_train = y_validate_feature
X_item_train, y_item_train = get_in_order(X_validate_feature_, X_validate_item_, accumulation_item, y_validate_item)


X_feature_test = accumulation_feature_test
y_feature_test = y_test_feature
X_item_test, y_item_test = get_in_order(X_test_feature_, X_test_item_, accumulation_item_test, y_test_item)



data_train_second = np.concatenate(([X_feature_train], [X_item_train], [accumulation_select_second_train]), axis=0).T
target_train_second = np.array(y_feature_train)
data_val_second = np.concatenate(([X_feature_test], [X_item_test], [accumulation_select_second_test]), axis=0).T
# data_val_second = np.concatenate(([X_feature_test], [X_item_test], [X_single1_test], [X_single2_test], [X_single3_test], [X_single4_test]), axis=0).T
target_val_second = np.array(y_feature_test)



b_size = 64
max_epochs = 30
num_iteration = 10
time_start = time.time()  # 开始计时
for ep in range(max_epochs-1, max_epochs):
    ba_sum = 0
    acc_sum = 0
    mse_min = 1000000
    for i in range(num_iteration):
        init = K.initializers.glorot_uniform(seed=1)
        simple_adam = K.optimizers.Adam()
        model = K.models.Sequential()
        model.add(K.layers.Dense(units=30, input_dim=3, activation='relu'))
        model.add(K.layers.Dense(units=15, input_dim=30, activation='relu'))
        model.add(K.layers.Dense(units=1))
        model.compile(loss='mse', optimizer=simple_adam, metrics=['mse'])
        h = model.fit(data_train_second, target_train_second, batch_size=b_size, epochs=ep + 1, shuffle=True, verbose=1)
        pred = model.predict(data_val_second)
        mse_val = mean_squared_error(target_val_second, pred)
        r2_val = r2_score(target_val_second, pred)
        print("mse validate: ", mse_val)
        print("r2 validate: ", r2_val)
        if mse_val < mse_min:
            mse_min = mse_val
    print(mse_min)