import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Lasso

def rmse(y_train, y_true):
    return np.sqrt(np.mean((np.log(y_train) - np.log(y_true))**2))
class KaggleModel(object):
    def __init__(self, trainFile, testFile):
        self._train = pd.read_csv(trainFile)
        self._test = pd.read_csv(testFile)
        self._testID = self._test.loc[:, "Id"]
        self._yTrain = self._train["SalePrice"]

    def clean(self):
        # Remove Id as it's not useful for prediction
        # print(self._test["Id"].shape, self._train["Id"].shape)
        self._test.drop(["Id"], axis=1, inplace=True)
        self._train.drop(["Id"], axis=1, inplace=True)

        # Remove outliers
        self._train = self._train.drop(self._train[(self._train['GrLivArea']>4000) & (self._train['SalePrice']<300000)].index)
        self._yTrain = self._train["SalePrice"]
        self._yTrain = np.log1p(self._yTrain)
        self.fill()
        

    def fill(self):
        all_data = pd.concat((self._train, self._test)).reset_index(drop=True)
        all_data.drop("SalePrice", axis=1, inplace=True)

        # Data descr says NA means "No pool"
        all_data["PoolQC"] = all_data["PoolQC"].fillna("None")
        # NA means "No misc features"
        all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")
        # NA means no alley access
        all_data["Alley"] = all_data["Alley"].fillna("None")
        # NA means no fence
        all_data["Fence"] = all_data["Fence"].fillna("None")
        # NA means no fireplace
        all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")
        # Assume neighorhoods have similar lot frontage
        all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
            lambda x: x.fillna(x.median()))
        # No garage specificities if there's no garage at all
        for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
            all_data[col] = all_data[col].fillna('None')
        # Same
        for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
            all_data[col] = all_data[col].fillna(0)
        # No basement
        for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
            all_data[col] = all_data[col].fillna(0)
        # Same
        for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
            all_data[col] = all_data[col].fillna('None')
        # No masonry veneer
        all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
        all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
        # Get Zoning from neighborhood
        all_data["MSZoning"] = all_data.groupby("Neighborhood")["MSZoning"].transform(
            lambda x: x.fillna(x.mode()))
        # Utilities is not varied enough
        all_data = all_data.drop(['Utilities'], axis=1)
        # NA means typical
        all_data["Functional"] = all_data["Functional"].fillna("Typ")
        # Only one NA
        all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
        all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
        all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
        all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
        all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
        all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")
        # That value is categorical
        all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
        all_data['OverallCond'] = all_data['OverallCond'].astype(str)
        all_data['YrSold'] = all_data['YrSold'].astype(str)
        all_data['MoSold'] = all_data['MoSold'].astype(str)
        # Adding new feature
        all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']

        cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
        for c in cols:
            lbl = LabelEncoder()
            lbl.fit(list(all_data[c].values))
            all_data[c] = lbl.transform(list(all_data[c].values))
        all_data = pd.get_dummies(all_data)
        pca = PCA(n_components=100)
        all_data = pca.fit_transform(all_data)
        self._train, self._test = (all_data[:len(self._train)], all_data[len(self._train):])

    def model(self):
        self._model = Lasso(alpha=0.0005)

    def test(self):
        x = self._train
        y = self._yTrain
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)
        # print(x_train.shape, y_train.shape)
        # print(y_train)
        self._model.fit(x_train, y_train)
        y_pred = self._model.predict(x_test)
        y_pred, y_test = np.expm1(y_pred), np.expm1(y_test)
        print(f"Accuracy is: {rmse(y_test, y_pred)}")

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
        x = self._train
        y = self._yTrain
        self._model.fit(x, y)

        x_test = self._test
        res_price = np.expm1(self._model.predict(x_test))
        res = self._testID.astype(int)
        resFrame = pd.DataFrame(
            data = {"Id": res, "SalePrice": res_price}
        )
        resFrame.to_csv("sub.csv", index=False)
        subNumber = self.__getSubNumber()
        os.system(f"git add -u && git commit -m \"Submission number {subNumber}\"")
        os.system(f"kaggle competitions submit -c house-prices-advanced-regression-techniques -f sub.csv -m \"Submission number {subNumber}\"")
