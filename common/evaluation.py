from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import copy


#%% Classification

def get_binary_cls_scores(y_real, y_pred):
    scores = {'recall': recall_score(y_real, y_pred),
              'precision': precision_score(y_real, y_pred),
              'f1': f1_score(y_real, y_pred),
              'accuracy': accuracy_score(y_real, y_pred)
              }
    return scores

# def get_multiple_cls_scores(y_real, y_pred, target_names):
#     scores = classification_report(y_real, y_pred, target_names=target_names, output_dict=True)
#     return scores

def get_multiple_cls_micro_score(y_real, y_pred):
    scores = {'recall': recall_score(y_real, y_pred, average='micro'),
              'precision': precision_score(y_real, y_pred, average='micro'),
              'f1': f1_score(y_real, y_pred, average='micro'),
              'accuracy': accuracy_score(y_real, y_pred)
              }
    return scores


def get_multiple_cls_macro_score(y_real, y_pred):
    scores = {'recall': recall_score(y_real, y_pred, average='macro'),
              'precision': precision_score(y_real, y_pred, average='macro'),
              'f1': f1_score(y_real, y_pred, average='macro'),
              'accuracy': accuracy_score(y_real, y_pred)
              }
    return scores

#%% Regression

def cal_r2(y, y_pred):
    if y.ndim!=1:
        y = y.ravel()
    if y_pred.ndim != 1:
        y_pred = y_pred.ravel()
    try:
        corr_val = np.corrcoef(y, y_pred)[0, 1]
        r2 = corr_val ** 2
    except:
        r2 = None

    return r2


def cal_rmse(y, y_pred):
    if y.ndim!=1:
        y = y.ravel()
    if y_pred.ndim != 1:
        y_pred = y_pred.ravel()
    return np.sqrt(mean_squared_error(y, y_pred))


def cal_residual(y, y_pred):
    return abs(y - y_pred)


def cal_mape(y, y_pred):
    y = y.reset_index(drop=True)
    idx = y[y!=0].index
    y = y[idx]
    y_pred = y_pred[idx]
    return np.mean(np.abs((y - y_pred)/y)) * 100


def cal_mape_temp(y, y_pred):
    y = y.reset_index(drop=True)
    idx = y[y!=0].index
    y = y[idx]
    y_pred = y_pred[idx]
    result = np.mean(np.abs((y - y_pred)/y)) * 100

    result = 0 if result > 100 else result

    return result



def cal_100_mape(y, y_pred):
    y = y.reset_index(drop=True)
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.reset_index(drop=True)
    idx = y[y!=0].index
    y = y[idx]
    y_pred = y_pred[idx]
    mae = np.abs((y - y_pred)/y)
    mae[mae > 1] = 1
    mape = np.mean(mae)
    result = (1 - mape) * 100
    return result


def cal_100_smape(y, y_pred, outlier=None):
    if isinstance(y, pd.Series):
        y = y.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values

    idx_zero = np.where(y != 0)[0]
    if outlier == None:
        idx_in = np.where(y)[0]
    else:
        idx_in = np.where(y < outlier)[0]
    idx = np.intersect1d(idx_zero, idx_in)

    y = y[idx]
    y_pred = y_pred[idx]
    smae = np.mean((np.abs(y - y_pred)) / (np.abs(y) + np.abs(y_pred)))
    result = (1 - smae) * 100
    return result


def cal_mae_list(y, y_pred):
    y = y.reset_index(drop=True)
    idx = y[y != 0].index
    y = y[idx]
    y_pred = y_pred[idx]
    return np.abs((y - y_pred) / y)


def cal_100_mase(y, y_pred):
    if isinstance(y, pd.Series):
        y = y.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values

    n = len(y)
    d = np.abs(np.diff(y)).sum() / (n-1)
    error = abs(y - y_pred)
    result = error.mean(axis=None)/d
    result = (1 -result) * 100
    return result