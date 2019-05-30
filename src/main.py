import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import math

train = pd.read_csv("train.csv")
train = pd.get_dummies(train).fillna(0)
test = pd.read_csv("test.csv")
print("First verision")
test.info()
test= pd.get_dummies(test).fillna(0)
print("Second version")
test.info()
print("Third version")
test.info()
test = test.loc[:, test.columns != "SalePrice"]
print("Fourth version")
test.info()
print(train.info())

x_train = train.loc[:, test.columns]
y_train = train.loc[:, "SalePrice"]

clf = RandomForestRegressor()
clf.fit(x_train, y_train)

x_test = test
y_test = clf.predict(x_test)
sub = test.loc[:, ["Id"]]
sub["SalePrice"] = y_test.astype(float)
sub.to_csv("Sub.csv", index=False)

y_true = train.loc[1200:, "SalePrice"]



def rmse(predictions, targets):
    return np.sqrt(np.mean((np.log(predictions)-np.log(targets))**2))

for i in range(len(y_test)):
    a, b = y_test[i], list(y_true)[i]
    print(a, b, b - a)

print(rmse(y_test, y_true))
