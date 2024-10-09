# 将两种训练方式得到的模型在验证集上得到的预测结果进行融合
# 融合涉及id对齐
# 元学习器选择一个简单的神经网络

from xgboost import XGBClassifier, XGBRegressor

from pandas import read_csv
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_squared_error, r2_score

# 读取数据
# 分裂属性的数据
df = read_csv("../../../dataset/test/CCPP_train.csv")
data = df.values
X_train_feature = data[:, 1:5].astype(int)
X_train_feature_ = data[:, 0:1].astype(int)
y_train_feature = data[:, -1].astype(int)

df = read_csv("../../../dataset/test/CCPP_validate.csv")
data = df.values
X_validate_feature = data[:, 1:5].astype(int)
X_validate_feature_ = data[:, 0:1].astype(int)
y_validate_feature = data[:, -1].astype(int)

df = read_csv("../../../dataset/test/CCPP_test.csv")
data = df.values
X_test_feature = data[:, 1:5].astype(int)
X_test_feature_ = data[:, 0:1].astype(int)
y_test_feature = data[:, -1].astype(int)


bst_feature = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=1, objective='reg:squarederror')
bst_feature.fit(X_train_feature, y_train_feature)
mse_feature = mean_squared_error(y_validate_feature, bst_feature.predict(X_validate_feature))
r2_feature = r2_score(y_validate_feature, bst_feature.predict(X_validate_feature))
accumulation_feature = bst_feature.predict(X_validate_feature, output_margin=True)
print("mse feature:", mse_feature)
print("r2 feature:", r2_feature)