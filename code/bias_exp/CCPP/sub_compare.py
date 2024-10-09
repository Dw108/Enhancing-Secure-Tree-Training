# 将两种训练方式得到的模型在验证集上得到的预测结果进行融合
# 融合涉及id对齐
# 元学习器选择一个简单的神经网络

from xgboost import XGBClassifier, XGBRegressor

from pandas import read_csv
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_squared_error, r2_score

# 读取数据
# 分裂属性的数据

df = read_csv("../../../dataset/CCPP/CCPP_sub_feature_train_enc.csv")
data = df.values
X_train_feature = data[:, 1:7].astype(int)
X_train_feature_ = data[:, 0:1].astype(int)
y_train_feature = data[:, -1].astype(int)

# df = read_csv("../../dataset/CCPP/CCPP_sub_feature_validate.csv")
df = read_csv("../../../dataset/CCPP/CCPP_sub_feature_validate_enc.csv")
data = df.values
X_validate_feature = data[:, 1:7].astype(int)
X_validate_feature_ = data[:, 0:1].astype(int)
y_validate_feature = data[:, -1].astype(int)

# df = read_csv("../../dataset/CCPP/CCPP_sub_feature_test.csv")
df = read_csv("../../../dataset/CCPP/CCPP_sub_feature_test_enc.csv")
data = df.values
X_test_feature = data[:, 1:7].astype(int)
X_test_feature_ = data[:, 0:1].astype(int)
y_test_feature = data[:, -1].astype(int)

# 分裂条目方式的数据
# df = read_csv("../../dataset/CCPP/CCPP_sub_item_train.csv")
df = read_csv("../../../dataset/CCPP/CCPP_sub_item_train_enc.csv")
data = df.values
X_train_item = data[:, 1:5]
X_train_item_ = data[:, 0:1]
y_train_item = data[:, -1]

# df = read_csv("../../dataset/CCPP/CCPP_sub_item_validate.csv")
df = read_csv("../../../dataset/CCPP/CCPP_sub_item_validate_enc.csv")
data = df.values
X_validate_item = data[:, 1:5]
X_validate_item_ = data[:, 0:1]
y_validate_item = data[:, -1]

# df = read_csv("../../dataset/CCPP/CCPP_sub_item_test.csv")
df = read_csv("../../../dataset/CCPP/CCPP_sub_item_test_enc.csv")
data = df.values
X_test_item = data[:, 1:5]
X_test_item_ = data[:, 0:1]
y_test_item = data[:, -1]

# 分裂属性的训练
bst_feature = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=1, objective='reg:squarederror')
bst_feature.fit(X_train_feature, y_train_feature)
mse_feature = mean_squared_error(y_validate_feature, bst_feature.predict(X_validate_feature))
r2_feature = r2_score(y_validate_feature, bst_feature.predict(X_validate_feature))
accumulation_feature = bst_feature.predict(X_validate_feature, output_margin=True)
print("mse feature:", mse_feature)
print("r2 feature:", r2_feature)

#
# 分裂条目的训练
bst_item = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=1, objective='reg:squarederror')
# fit model
bst_item.fit(X_train_item, y_train_item)
# make predictions
mse_item = mean_squared_error(y_validate_item, bst_item.predict(X_validate_item))
r2_item = r2_score(y_validate_item, bst_item.predict(X_validate_item))
accumulation_item = bst_item.predict(X_validate_item, output_margin=True)
print("mse item:", mse_item)
print("r2 item:", r2_item)
#
# dict_tmp = {}
# for i in range(accumulation_feature.shape[0]):
#     dict_tmp.update({int(X_validate_feature_[i][0]): accumulation_feature[i]})
# data_train_second = []
# label_train_second = []
# for i in range(accumulation_item.shape[0]):
#     # id
#     arr1 = dict_tmp[int(X_validate_item_[i][0])]
#     arr2 = accumulation_item[i]
#     tmp = numpy.concatenate((arr1, arr2), axis=0)
#     data_train_second.append(tmp)
#     label_train_second.append(y_validate_item[i])
#
# accumulation_feature_test_second = bst_feature.predict(X_test_feature, output_margin=True)
# accumulation_item_test_second = bst_item.predict(X_test_item, output_margin=True)
# dict_tmp = {}
# for i in range(accumulation_feature_test_second.shape[0]):
#     dict_tmp.update({int(X_test_feature_[i][0]): accumulation_feature_test_second[i]})
# data_val_second = []
# label_val_second = []
# for i in range(accumulation_item_test_second.shape[0]):
#     # id
#     arr1 = dict_tmp[int(X_test_item_[i][0])]
#     arr2 = accumulation_item_test_second[i]
#     tmp = numpy.concatenate((arr1, arr2), axis=0)
#     data_val_second.append(tmp)
#     label_val_second.append(y_test_item[i])

#
# data_train_second = np.array(data_train_second)
# label_train_second = np.array(label_train_second)
# data_val_second = np.array(data_val_second)
# label_val_second = np.array(label_val_second)
#
#
#
# b_size = 32
# max_epochs = 30
# num_iteration = 10
# for ep in range(max_epochs):
#     ba_sum = 0
#     acc_sum = 0
#     for i in range(num_iteration):
#         # 2. 定义模型
#         init = K.initializers.glorot_uniform(seed=1)
#         simple_adam = K.optimizers.Adam()
#         model = K.models.Sequential()
#         model.add(K.layers.Dense(units=14, input_dim=14, kernel_initializer=init, activation='relu'))
#         model.add(K.layers.Dense(units=10, kernel_initializer=init, activation='relu'))
#         model.add(K.layers.Dense(units=7, kernel_initializer=init, activation='softmax'))
#         model.compile(loss='sparse_categorical_crossentropy', optimizer=simple_adam, metrics=['accuracy'])
#         print("iteration:", i)
#         print("Starting training ")
#         h = model.fit(data_train_second, label_train_second, batch_size=b_size, epochs=ep+1, shuffle=True, verbose=1)
#         print("Training finished \n")
#
#         # 4. 评估模型
#         eval = model.evaluate(data_val_second, label_val_second, verbose=0)
#         print("Evaluation on test data: loss = %0.6f accuracy = %0.2f%% \n" \
#               % (eval[0], eval[1] * 100))
#         ba = balanced_accuracy_score(label_val_second, numpy.argmax(model.predict(data_val_second), axis=1))
#         print("balanced acc:", ba)
#         ba_sum = ba_sum + ba
#         acc = accuracy_score(label_val_second, numpy.argmax(model.predict(data_val_second), axis=1))
#         print("acc:", acc)
#         acc_sum = acc_sum + acc
#     mean_ba = ba_sum/num_iteration
#     mean_acc = acc_sum / num_iteration
#     print("epochs:", ep + 1, "mean balanced acc:", mean_ba)
#     print("epochs:", ep + 1, "mean acc:", mean_acc)
