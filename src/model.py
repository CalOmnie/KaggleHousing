import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Lasso, ElasticNet, BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from scipy.stats import skew
from scipy.special import boxcox1p
import lightgbm as lgb
import xgboost as xgb

np.random.seed(0)

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
        # self.augment()

        self._yTrain = self._train.SalePrice.values
        self._yTrain = np.log1p(self._yTrain)
        self.fill()
        self.addNew()
        self.handleNumerical()
        self.handleCategorical()

    def fill(self):
        all_data = pd.concat((self._train, self._test), sort=False).reset_index(drop=True)
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
        self._train, self._test = (all_data[:len(self._train)], all_data[len(self._train):])

    def addNew(self):
        all_data = pd.concat((self._train, self._test)).reset_index(drop=True)
        # Adding new feature
        all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['GrLivArea']
        all_data["AllBath"] = all_data["FullBath"] + 0.5*all_data["HalfBath"] + all_data["BsmtFullBath"] + 0.5*all_data["BsmtHalfBath"]
        all_data["Remod"] = (all_data["YearBuilt"] == all_data["YearRemodAdd"]).astype(int)
        all_data["Age"] = all_data["YrSold"] - all_data["YearRemodAdd"]
        all_data["IsNew"] = (all_data["YrSold"] == all_data["YearBuilt"]).astype(int)
        neighborhoods = {
            "StoneBr":2,
            "NridgHt":2,
            "NoRidge":2,
            "Meadow":0,
            "IDOTRR":0,
            "BrDale":0,
        }
        all_data["NeighRich"] = all_data["Neighborhood"].apply(lambda x: neighborhoods.get(x, 1))
        all_data["TotalPorchSF"] = (all_data["OpenPorchSF"] + all_data["EnclosedPorch"] +
                                    all_data["3SsnPorch"] + all_data["ScreenPorch"])
        toDrop = ['YearRemodAdd', 'GarageYrBlt', 'GarageArea', 'GarageCond', 'TotalBsmtSF', 'TotRmsAbvGrd', 'BsmtFinSF1']
        all_data.drop(toDrop, axis=1, inplace=True)
        self._train, self._test = (all_data[:len(self._train)], all_data[len(self._train):])

    def handleCategorical(self):
        all_data = pd.concat((self._train, self._test)).reset_index(drop=True)
        # Add extra categorical values
        all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
        all_data['OverallCond'] = all_data['OverallCond'].astype(str)
        all_data['YrSold'] = all_data['YrSold'].astype(str)
        all_data['MoSold'] = all_data['MoSold'].astype(str)
        cat_feats = all_data.dtypes[all_data.dtypes == "object"].index
        num_feats = all_data.dtypes[all_data.dtypes != "object"].index
        encoded = pd.get_dummies(all_data[cat_feats])

        encoded_test = encoded[len(self._train):]
        zeros = encoded_test.sum() == 0
        encoded.drop(encoded_test.columns[zeros], axis=1, inplace=True)
        encoded_train = encoded[len(self._train):]
        zeros = encoded_train.sum() < 10
        encoded.drop(encoded_train.columns[zeros], axis=1, inplace=True)

        all_data = pd.concat((all_data[num_feats], encoded), axis=1)
        self._train, self._test = (all_data[:len(self._train)], all_data[len(self._train):])

    def handleNumerical(self):
        all_data = pd.concat((self._train, self._test)).reset_index(drop=True)
        numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
        # Check the skew of all numerical features
        skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x)).sort_values(ascending=False)
        skewness = pd.DataFrame({'Skew': skewed_feats})
        skewness = skewness.loc[skewness["Skew"] > 0.75]
        skewed_features = skewness.index
        lam = 0.15
        for feat in skewed_features:
            all_data[feat] = boxcox1p(all_data[feat], lam)

        self._train, self._test = (all_data[:len(self._train)], all_data[len(self._train):])

    def pca(self):
        all_data = pd.concat((self._train, self._test)).reset_index(drop=True)
        pc = PCA()
        all_data = pc.fit_transform(all_data)
        self._train, self._test = (all_data[:len(self._train)], all_data[len(self._train):])

    def augment(self):
        augmented = []
        for _, row in self._train.iterrows():
            newRow = row.copy()
            noise = row["SalePrice"] / 20
            numAug = 1
            for i in range(numAug):
                newRow["SalePrice"] += random.uniform(-noise, noise)
                augmented.append(newRow)
        self._train = self._train.append(augmented)

    def model(self):
        lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
        ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
        KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
        model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                                     learning_rate=0.05, max_depth=3,
                                     min_child_weight=1.7817, n_estimators=2200,
                                     reg_alpha=0.4640, reg_lambda=0.8571,
                                     subsample=0.5213, silent=1,
                                     random_state=7, nthread =-1)
        GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state =5)
        model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                                              learning_rate=0.05, n_estimators=720,
                                              max_bin = 55, bagging_fraction = 0.8,
                                              bagging_freq = 5, feature_fraction = 0.2319,
                                              feature_fraction_seed=9, bagging_seed=9,
                                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
        stacked_model = StackingAveragedModels(base_models=(ENet, GBoost, KRR), meta_model=lasso)
        self._models = (lasso, ENet, KRR, GBoost, stacked_model)

    def _rmse_cv(self):
        n_folds = 5
        kf = KFold(n_folds, shuffle=False, random_state=42).get_n_splits(self._train.values)
        res = []
        for model in self._models:
            res.append(np.sqrt(-cross_val_score(model,
                                            self._train.values,
                                            self._yTrain,
                                            scoring="neg_mean_squared_error",
                                            cv=kf)))
        return res

    def test(self):
        res = self._rmse_cv()
        for r in res:
            print(f"Accuracy is: {r.mean()}, {r.std()}")

    def __getSubNumber(self):
        with open("submissions", "r+") as f:
            val = f.read()
            if not val:
                newVal = 1
            else:
                newVal = int(val) + 1
            f.seek(0)
            f.write(str(newVal))
            f.truncate()
            return newVal

    def submit(self):
        x = self._train.to_numpy()
        y = self._yTrain
        model = self._models[-1]
        print(x.shape, y.shape)
        model.fit(x, y)

        x_test = self._test
        res_price = np.expm1(model.predict(x_test))
        res = self._testID.astype(int)
        resFrame = pd.DataFrame(
            data = {"Id": res, "SalePrice": res_price}
        )
        resFrame.to_csv("sub.csv", index=False)
        subNumber = self.__getSubNumber()
        os.system(f"git add -u && git commit -m \"Submission number {subNumber}\"")
        os.system(f"kaggle competitions submit -c house-prices-advanced-regression-techniques -f sub.csv -m \"Submission number {subNumber}\"")



class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    #Do the predictions of all base models on the test data and use the averaged predictions as 
    #meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)
