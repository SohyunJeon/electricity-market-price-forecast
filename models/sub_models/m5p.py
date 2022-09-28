import numpy as np
import pandas as pd
import random
random.seed(42)

import xgboost
import optuna
from optuna import Trial
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_val_score
from m5py import M5Prime
import m5py
from common import evaluation

from models.fps_model import SubModel

'''
xgb = XGBRegressor(max_depth=1,
             eta=0.1,
             subsample=0.9,
             min_child_weight=1)
xgb.fit(train_X, train_y, eval_set=[(valid_X, valid_y)],
        # nrounds=200,
        early_stopping_rounds=10,
        # print_every_n=5
        )
        '''

class M5PSubModel(SubModel):
    def __init__(self, name='M5Prime', version=m5py.__version__, estimator=M5Prime()):
        super().__init__(name=name,
                         version=version,
                         estimator=estimator
                         )



    def objective(self, trial:Trial, X: pd.DataFrame, y: pd.Series, cv, tunning_scoring):
        params = {
            'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
            'subsample': trial.suggest_loguniform('subsample', 0.8, 1),
            'eta': trial.suggest_loguniform('eta', 0.05, 0.3),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 3),
            # 'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
            # 'colsample_bytree': trial.suggest_categorical('colsample_bytree',
            #                                               [0.8, 0.9, 1.0]),
            # 'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.5),
            # 'n_estimators': trial.suggest_categorical('n_estimators', [100, 500, 700]),
            # 'max_depth': trial.suggest_int('max_depth', 4, 10),
            # 'random_state': 42,
            # 'min_child_weight': trial.suggest_int('min_child_weight', 1, 300)
                    }
        model = self.estimator.set_params(**params)
        result = cross_val_score(model, X, y, cv=cv, scoring=tunning_scoring)
        print('result : ', result)
        return np.mean(result)

    def custom_criteria(self, pred, dtrain):
        y = dtrain.get_label()
        idx = np.where(y != 0)[0]
        y = y[idx]
        y_pred = pred[idx]
        mae = np.abs((y - y_pred) / y)
        mae[mae > 1] = 1
        mape = np.mean(mae)
        # result = (1 - mape) * 100
        result = mape * 100
        return 'custom_eval', result

    def fit_predict(self, X: pd.DataFrame, y: pd.Series, outlier: int=100,
                    n_tunning: int=5, tunning_scoring: str='neg_root_mean_squared_error')-> np.array:
        te_i = len(X) - 24
        train_X, test_X = X.iloc[:te_i, :].reset_index(drop=True), X.iloc[te_i:, :].reset_index(drop=True)
        train_y, test_y = y[:te_i].reset_index(drop=True), y[te_i:].reset_index(drop=True)

        # get score
        self.model = M5Prime()
        self.model.fit(X, y)

        yhat = self.model.predict(test_X)
        self.scores = {
            'rmse': evaluation.cal_rmse(test_y, yhat),
            'r2': evaluation.cal_r2(test_y, yhat),
            '100-mape': evaluation.cal_100_mape(test_y, yhat),
            '100-smape': evaluation.cal_100_smape(test_y, yhat, outlier),
        }

        print('Scores : ', self.scores)
        return yhat

    def set_feature_importance(self, X: pd.DataFrame):
        select_limit = 1e-2
        importance = pd.Series(data=self.model.feature_importances_,
                               index=X.columns)
        importance = importance.sort_values(ascending=False)
        self.feature_importance = importance[importance > select_limit]
