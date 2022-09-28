
from datetime import timedelta
from pytz import timezone
import dateutil
from dateutil.parser import parse
import random
random.seed(42)

import models
import importlib
importlib.reload(models)
from models.fps_model import StackedModel
from models.sub_models.xgboost import XGBoostRegressorSubModel
from models.sub_models.m5p import M5PSubModel
from qpslib_model_manager import model_client
from models import model_save
from models.conn_obj import DBConn
from models.preprocess import features
from models.conf import config, ModelParams

#%% Service Setting

company = 'EIP'
# target = 'ERCOT_HOUSTON'
target = 'MVP_ERCOT_HOUSTON'

host = 'esbr-eip.brique.kr:8000'
result_type ='REGRESSION'
client = model_client.FPSModelClient(host, company, target, result_type)

local_tz = timezone(config['timezone'])

#%% Model Setting
time_param = ModelParams.local_date
train_days = ModelParams.train_days
train_start = '2022-03-01 00:00:00'
# train_end = (dateutil.parser.parse(train_start) + timedelta(days=train_days)).strftime('%Y-%m-%d %H:%M:%S')
train_end = '2022-04-01 00:00:00'

#%% Load features

conn = DBConn()
collection = conn.esbr_summ_db[target]
summary_df = features.load_features_with_duration(collection, time_param, train_start, train_end)


#%% Split data

use_params = ModelParams.total_pow_params

train_X = summary_df.loc[:, use_params].astype(float)
train_y = summary_df['y']



#%% Sub-model modeling

# xgb_sub = XGBoostRegressorSubModel()
# xgb_yhat = xgb_sub.fit_predict(train_X, train_y)
# xgb_sub.set_feature_importance(X=train_X)

train_X_fillna = train_X.fillna(0)

outlier = train_y.mean()
sub = M5PSubModel()
sub_yhat = sub.fit_predict(train_X_fillna, train_y, outlier)
sub.set_feature_importance(X=train_X_fillna)


#%% define model
sub_models = {
    sub.name: sub,
              }
stacked_model = StackedModel(meta_model=sub,
                             sub_models=sub_models,
                             preprocessing=None,
                             features=use_params
                             )
sub_models_name = f'Submodels : {len(sub_models.keys())}'
stacked_model_name = 'EIP MVP'
meta = {'outlier': outlier}


#%% Model save
sub_save_resp = model_save.save_sub_models(client=client,
                                           sub_models=sub_models,
                                           name=sub_models_name)

stacked_save_resp = model_save.save_stacked_model(client=client,
                                                  stacked_model=stacked_model,
                                                  ref_id=sub_save_resp.id,
                                                  score=stacked_model.meta_model.scores,
                                                  name=stacked_model_name,
                                                  meta=meta,
                                                feats=stacked_model.meta_model.feature_importance)
client.set_best_model(stacked_save_resp.id)