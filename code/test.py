from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from pandas import read_csv
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
data = load_iris()
# X_train, X_test, y_train, y_test = train_test_split(data['data'], data['target'], test_size=.2)

df = read_csv("../dataset/covtype/covtype_sub_item_train.csv")
data = df.values
# X_train = data[:, 0:12]
X_train = data[:, 1:13]
y_train = data[:, -1] -1

df = read_csv("../dataset/covtype/covtype_sub_item_test.csv")
data = df.values
# X_test = data[:, 0:12]

X_test = data[:, 1:13]
y_test = data[:, -1] -1
acc = []
# for i in range(300):
# create model instance
bst = XGBClassifier(n_estimators=300, max_depth=8, learning_rate=1, objective='multi:softprob')
# fit model
bst.fit(X_train, y_train)
# make predictions
tmp = accuracy_score(y_test, bst.predict(X_test))
acc.append(tmp)
print(acc)
# print("iteration:", i, "  acc:", tmp)
# prob = bst.predict(X_test, output_margin=True)
# res = bst.predict(X_test)



# x_axis_data = list(range(0, 300))  # x
# y_axis_data = acc  # y
#
# plt.plot(x_axis_data, y_axis_data, 'b*--', alpha=0.5, linewidth=1, label='acc')  # 'bo-'表示蓝色实线，数据点实心原点标注
# ## plot中参数的含义分别是横轴值，纵轴值，线的形状（'s'方块,'o'实心圆点，'*'五角星   ...，颜色，透明度,线的宽度和标签 ，
#
# plt.legend()  # 显示上面的label
# plt.xlabel('n_estimators')  # x_label
# plt.ylabel('acc')  # y_label
#
# # plt.ylim(-1,1)#仅设置y轴坐标范围
# plt.show()
