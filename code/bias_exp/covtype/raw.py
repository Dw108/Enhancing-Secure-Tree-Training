# 将两种训练方式得到的模型在验证集上得到的预测结果进行融合
# 融合涉及id对齐
# 元学习器选择一个简单的神经网络

from xgboost import XGBClassifier

from pandas import read_csv
from sklearn.metrics import accuracy_score, balanced_accuracy_score

# 读取数据
# 分裂属性的数据
# df = read_csv("../../../dataset/test/covtype_train.csv")
df = read_csv("../../../dataset/test/covtype_train_enc.csv")
data = df.values
X_train_feature = data[:, 1:13].astype(int)
X_train_feature_ = data[:, 0:1].astype(int)
y_train_feature = data[:, -1].astype(int) - 1

# df = read_csv("../../../dataset/test/covtype_validate.csv")
df = read_csv("../../../dataset/test/covtype_validate_enc.csv")
data = df.values
X_validate_feature = data[:, 1:13].astype(int)
X_validate_feature_ = data[:, 0:1].astype(int)
y_validate_feature = data[:, -1].astype(int) - 1

# df = read_csv("../../../dataset/test/covtype_test.csv")
df = read_csv("../../../dataset/test/covtype_test_enc.csv")
data = df.values
X_test_feature = data[:, 1:13].astype(int)
X_test_feature_ = data[:, 0:1].astype(int)
y_test_feature = data[:, -1].astype(int) - 1

# 分裂属性的训练
bst_feature = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=1, objective='multi:softprob')
bst_feature.fit(X_train_feature, y_train_feature)
acc_feature = accuracy_score(y_validate_feature, bst_feature.predict(X_validate_feature))
b_acc_feature = balanced_accuracy_score(y_validate_feature, bst_feature.predict(X_validate_feature))
accumulation_feature = bst_feature.predict(X_validate_feature, output_margin=True)
print("acc feature:", acc_feature)
print("acc balanced feature:", b_acc_feature)