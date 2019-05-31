import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def rmse(predictions, targets):
    return np.sqrt(np.mean((np.log(predictions)-np.log(targets))**2))


class KaggleModel(object):
    def __init__(self, trainFile, testFile):
        self._train = pd.read_csv(trainFile)
        self._test = pd.read_csv(testFile)

    def clean(self):
        self._train = pd.get_dummies(self._train).fillna(0)
        self._test = pd.get_dummies(self._test).fillna(0)

    def model(self):
        self._model = RandomForestRegressor()

    def test(self):
        x = self._train[self._test.columns]
        y = self._train["SalePrice"]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)
        # print(x_train.shape, y_train.shape)
        # print(y_train)
        self._model.fit(x_train, y_train)
        y_pred = self._model.predict(x_test)
        print(f"Accuracy is: {rmse(y_pred, y_test)}")

    def __getSubNumber(self):
        with open("submissions", "r+") as f:
            val = f.read()
            print(val)
            if not val:
                newVal = 1
            else:
                newVal = int(val) + 1
            f.seek(0)
            f.write(str(newVal))
            f.truncate()
            return newVal

    def submit(self):
        x = self._train[self._test.columns]
        y = self._train["SalePrice"]
        self._model.fit(x, y)
        x_test = self._test
        res_price = self._model.predict(x_test)
        res = self._test["Id"]
        res["SalePrice"] = res_price
        res.to_csv("sub.csv", index=False)
        subNumber = self.__getSubNumber()
        os.system(f"git add -u && git commit -m \"Submission number {subNumber}\"")
        os.system(f"kaggle competitions submit -c house-prices-advanced-regression-techniques -f sub.csv -m \"Submission number {subNumber}\"")
