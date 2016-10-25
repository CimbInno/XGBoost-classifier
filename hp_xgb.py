## Multiclass Classification with the Hidden Preferred dataset
import pandas
import pickle
import numpy as np
from pandas import read_csv
from numpy import loadtxt
from numpy import sort
from xgboost import XGBClassifier
from xgboost import plot_tree
from xgboost import plot_importance
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from matplotlib import pyplot


## Load training/test and production datasets
dataframe = pandas.read_csv("hp500.csv", delimiter=",", header=None)
dataframe = dataframe.fillna(0)
#print dataframe
dataset = dataframe.values
X = dataset[:,1:100].astype(float)
Y = dataset[:,100]
dataframe2 = pandas.read_csv("hp1000.csv", delimiter=",", header=None)
dataframe2 = dataframe2.fillna(0)
#print dataframe2
dataset2 = dataframe2.values
X2 = dataset2[:,1:100].astype(float)
#X2 = dataset2[:,1:100].astype(dtype=np.str)

## Encode string class values as integers
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(Y)
label_encoded_y = label_encoder.transform(Y)
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, label_encoded_y, test_size=test_size, random_state=seed)

## Build, train, validate/test and run predictions on XGBoost model
model = XGBClassifier()

kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
kfold2 = StratifiedKFold(n_splits=9, shuffle=True, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold)
results2 = cross_val_score(model, X, Y, cv=kfold2)
print("CV Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
print("Stratified CV Accuracy: %.2f%% (%.2f%%)" % (results2.mean()*100, results2.std()*100))

model.fit(X_train, y_train)
print("XGBoost model:")
print(model)

#predictions_class = [round(value) for value in X2_pred]
predictions_class = model.predict(X2)
predictions_prob = model.predict_proba(X2)

#accuracy = accuracy_score(y_test, predictions)
#print("Accuracy: %.2f%%" % (accuracy * 100.0))

print("Predictions:")
print(predictions_class)
print(predictions_prob)
predictions_out = label_encoder.inverse_transform(predictions_class)
print(predictions_out)

## Output predictions to a file
counter = 0
f = open("predictions_xgb1.tmp", mode="w")
for item in predictions_out:
	counter += 1
	f.write(str(counter) + "," + str(item) + "\n")
f.close()

counter = 0
f = open("predictions_xgb2.tmp", mode="w")
for line in open("hp1000.csv"):
	columns = line.split(",")
	counter += 1
	f.write(str(counter) + "," + str(columns[0]) + "\n")
f.close()

a_cols = ['id', 'prediction']
a = pandas.read_csv('predictions_xgb1.tmp', sep=',', names=a_cols, encoding='latin-1')
b_cols = ['id', 'cardno']
b = pandas.read_csv('predictions_xgb2.tmp', sep=',', names=b_cols, encoding='latin-1')
merged = b.merge(a, on='id')
merged.to_csv("predictions_xgb.txt", index=False)

## Ensembling Xgb and Dnn predictions to a file
a_cols = ['id','cardno','pred_xgb']
a = pandas.read_csv('predictions_xgb.txt', sep=',', names=a_cols, encoding='latin-1')
b_cols = ['id','cardno','pred_dnn']
b = pandas.read_csv('predictions_dnn.txt', sep=',', names=b_cols, encoding='latin-1')
merged = b.merge(a, on=('id', 'cardno'))
merged.to_csv("predictions.txt", index=False)
#c_cols = ['id','cardno','pred1','pred2']
#c = pandas.read_csv('predictions.txt', sep=',', names=a_cols, encoding='latin-1')

## Plotting model trees
plot_tree(model)
pyplot.show()

## Printing and plotting feature importance scores
#print(model.feature_importances_)
#pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
#pyplot.show()
plot_importance(model)
pyplot.show()

## Saving model to a file
print("Saving model to: hp_xgb.mdl")
pickle.dump(model, open("hp_xgb.mdl", "wb"))


## Fitting a model using each feature importance score as a threshold
thresholds = sort(model.feature_importances_)
for thresh in thresholds:
	# Select features using threshold
	selection = SelectFromModel(model, threshold=thresh, prefit=True)
	select_X_train = selection.transform(X_train)
	# Train model
	selection_model = XGBClassifier()
	selection_model.fit(select_X_train, y_train)
	# Evaluate model
	select_X_test = selection.transform(X_test)
	y_pred = selection_model.predict(select_X_test)
	predictions = [round(value) for value in y_pred]
	accuracy = accuracy_score(y_test, predictions)
	print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))


