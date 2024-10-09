# use feature importance for feature selection
import copy

from numpy import loadtxt
from numpy import sort
from pandas import read_csv
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel

df_train = read_csv("../dataset/overlappingnums/covtype/covtype_sub_feature_train.csv")
# df_train = read_csv("./dataset/overlappingnums/covtype/covtype_sub_feature_train_enc.csv")
data = df_train.values
X_train = data[:, 1:5].astype(int)
y_train = data[:, -1].astype(int) - 1

df_test = read_csv("../dataset/overlappingnums/covtype/covtype_sub_feature_validate.csv")
# df_test = read_csv("./dataset/overlappingnums/covtype/covtype_sub_feature_validate_enc.csv")
data = df_test.values
X_test = data[:, 1:5].astype(int)
y_test = data[:, -1].astype(int) - 1


# fit model on all training data
model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=1, objective='multi:softprob')
model.fit(X_train, y_train)
# make predictions for test data and evaluate
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
# Fit model using each importance as a threshold
thresholds = sort(model.feature_importances_)
max_acc = 0
best_model = 0
best_X = 0
for thresh in thresholds:
    # select features using threshold
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    select_X_train = selection.transform(X_train)
    # train model
    selection_model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=1, objective='multi:softprob')
    selection_model.fit(select_X_train, y_train)
    # eval model
    select_X_test = selection.transform(X_test)
    y_pred = selection_model.predict(select_X_test)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)
    if accuracy > max_acc:
        max_acc = accuracy
        best_model = copy.deepcopy(selection_model)
        best_X = copy.deepcopy(select_X_test)
    print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy * 100.0))

y_pred = best_model.predict(best_X)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, best_X.shape[1], accuracy * 100.0))