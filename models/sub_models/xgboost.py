import numpy as np
import pandas as pd

import xgboost
import optuna
from optuna import Trial
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
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

class XGBoostRegressorSubModel(SubModel):
    def __init__(self, name='XGBoost_Regressor', version=xgboost.__version__, estimator=xgboost.XGBRegressor(),
                 stop_limit=10):
        super().__init__(name=name,
                         version=version,
                         estimator=estimator
                         )
        self.stop_limit = stop_limit


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

    def fit_predict(self, X: pd.DataFrame, y: pd.Series, n_tunning:int=5, tunning_scoring:str='neg_root_mean_squared_error')-> np.array:
        iter_set = 8000
        te_i = len(X) - 24
        train_X, test_X = X.iloc[:te_i, :], X.iloc[te_i:, :]
        train_y, test_y = y[:te_i], y[te_i:]

        model_params = {'max_depth': 1,
                        'eta': 0.1,
                        'subsample': 0.9,
                        'min_child_weight': 1,
                        'n_estimators': iter_set,
                        'random_state': 42}
        model = XGBRegressor(**model_params)

        early_stopping = xgboost.callback.EarlyStopping(rounds=iter_set)
        fit_params = {'callbacks': [early_stopping],
                      'eval_metric': self.custom_criteria,
                      'eval_set': [(train_X, train_y)]
                      }
        model.fit(train_X, train_y,
                  **fit_params,
                  verbose=False)

        # early stopping 조건 변경
        # stop_limit = (train_y.std())/30

        valid_scores = np.array(model.evals_result()['validation_0']['custom_eval'])
        # success_scores = np.where(valid_scores < stop_limit)[0]
        success_scores = np.where(valid_scores < self.stop_limit)[0]
        if len(success_scores) == 0:
            iter_rounds = iter_set
        else:
            iter_rounds = success_scores[0]
        print('iter_rounds : ', iter_rounds)
        model_params['n_estimators'] = iter_rounds

        self.model = XGBRegressor(**model_params)
        self.model.fit(X, y,
                  **fit_params,
                  verbose=False)

        yhat = self.model.predict(test_X)

        self.scores = {
            'rmse': evaluation.cal_rmse(test_y, yhat),
            'r2': evaluation.cal_r2(test_y, yhat),
            '100-mape': evaluation.cal_100_mape(test_y, yhat)
        }

        print('Scores : ', self.scores)
        return yhat

    #
    #
    #
    # def fit_predict_old(self, X: pd.DataFrame, y: pd.Series, n_tunning:int=5, tunning_scoring:str='neg_root_mean_squared_error')-> np.array:
    #     iter_set = 8000
    #     te_i = len(X) - 24
    #     train_X, test_X = X.iloc[:te_i, :], X.iloc[te_i:, :]
    #     train_y, test_y = y[:te_i], y[te_i:]
    #
    #     model_params = {'max_depth': 1,
    #                     'eta': 0.1,
    #                     'subsample': 0.8,
    #                     'min_child_weight': 0.8,
    #                     'n_estimators': iter_set,
    #                     'random_state': 42}
    #     model = XGBRegressor(**model_params)
    #
    #     early_stopping = xgboost.callback.EarlyStopping(rounds=iter_set)
    #     fit_params = {'callbacks': [early_stopping],
    #                   'eval_metric': self.custom_criteria,
    #                   'eval_set': [(train_X, train_y)]
    #                   }
    #     model.fit(train_X, train_y,
    #               **fit_params,
    #               verbose=False)
    #
    #     # early stopping 조건 변경
    #     # stop_limit = (train_y.std())/30
    #     stop_limit = 30
    #     valid_scores = np.array(model.evals_result()['validation_0']['custom_eval'])
    #     # success_scores = np.where(valid_scores < stop_limit)[0]
    #     success_scores = np.where(valid_scores < stop_limit)[0]
    #     if len(success_scores) == 0:
    #         iter_rounds = iter_set
    #     else:
    #         iter_rounds = success_scores[0]
    #     print('iter_rounds : ', iter_rounds)
    #     model_params['n_estimators'] = iter_rounds
    #
    #     self.model = XGBRegressor(**model_params)
    #     self.model.fit(train_X, train_y,
    #               **fit_params,
    #               verbose=False)
    #
    #     yhat = self.model.predict(test_X)
    #
    #     self.scores = {
    #         'rmse': evaluation.cal_rmse(test_y, yhat),
    #         'r2': evaluation.cal_r2(test_y, yhat),
    #         '100-mape': evaluation.cal_100_mape(test_y, yhat)
    #     }
    #     # 최종 모델 fit
    #     self.model.fit(X, y,
    #                    **fit_params,
    #                    verbose=False
    #                    )
    #     print('Scores : ', self.scores)
    #     return yhat
    #
    #
    #
    # def fit_predict_old1(self, X: pd.DataFrame, y: pd.Series, n_tunning:int=5, tunning_scoring:str='neg_root_mean_squared_error')-> np.array:
    #     te_i = len(X) - 24
    #     train_X, test_X = X.iloc[:te_i, :], X.iloc[te_i:, :]
    #     train_y, test_y = y[:te_i], y[te_i:]
    #
    #     self.model = XGBRegressor(max_depth=2,
    #                      eta=0.2,
    #                      subsample=0.9,
    #                      min_child_weight=1,
    #                      verbosity=None,
    #                      n_estimators=500,
    #                     random_state=42
    #
    #                     )
    #
    #     self.model.fit(train_X, train_y,
    #                    # verbose
    #             # nrounds=200,
    #
    #             # early_stopping_rounds=10,
    #             # print_every_n=5
    #                    # y 편차 1/10 < rmse -> 이때 멈추기
    #
    #             )
    #
    #     yhat = self.model.predict(test_X)
    #
    #     self.scores = {
    #         'rmse': evaluation.cal_rmse(test_y, yhat),
    #         'r2': evaluation.cal_r2(test_y, yhat),
    #         '100-mape': 100 - evaluation.cal_mape(test_y, yhat)
    #     }
    #     # 최종 모델 fit
    #     self.model.fit(X, y)
    #     print('Scores : ', self.scores)
    #     return yhat
    #
    #
    #
    #
    #
    # def fit_predict_old2(self, X: pd.DataFrame, y: pd.Series, n_tunning:int=5, tunning_scoring:str='neg_root_mean_squared_error')-> np.array:
    #     self.model = XGBRegressor(max_depth=1,
    #                        eta=0.1,
    #                        subsample=0.9,
    #                        min_child_weight=1)
    #     te_i = len(X) - 24
    #     train_X, test_X = X.iloc[:te_i, :], X.iloc[te_i:, :]
    #     train_y, test_y = y[:te_i], y[te_i:]
    #     # tr_i = int(len(X)*0.7)
    #     # te_i = tr_i + int(len(X)*0.2)
    #     # train_X, valid_X, test_X = X.iloc[:tr_i, :], X.iloc[tr_i:te_i, :], X.iloc[te_i:, :]
    #     # train_y, valid_y, test_y = y[:tr_i], y[tr_i:te_i], y[te_i:]
    #     # self.model.fit(train_X, train_y, eval_set=[(valid_X, valid_y)],
    #     #         # nrounds=200,
    #     #         # early_stopping_rounds=10,
    #     #         # print_every_n=5
    #     #         )
    #     self.model.fit(train_X, train_y,
    #             # nrounds=200,
    #             # early_stopping_rounds=10,
    #             # print_every_n=5
    #             )
    #
    #     yhat = self.model.predict(test_X)
    #
    #     self.scores = {
    #         'rmse': evaluation.cal_rmse(test_y, yhat),
    #         'r2': evaluation.cal_r2(test_y, yhat),
    #         '100-mape': 100 - evaluation.cal_mape(test_y, yhat)
    #     }
    #     # 최종 모델 fit
    #     self.model.fit(X, y)
    #     print('Scores : ', self.scores)
    #     return yhat

    def set_feature_importance(self, X: pd.DataFrame):
        select_limit = 1e-2
        importance = pd.Series(data=self.model.feature_importances_,
                               index=X.columns)
        importance = importance.sort_values(ascending=False)
        self.feature_importance = importance[importance > select_limit]
