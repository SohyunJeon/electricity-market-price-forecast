import timeit
from types import SimpleNamespace
import pandas as pd
import dill as dill
import json
import traceback
from datetime import datetime, timedelta
from models import model_save
from models.sub_models.xgboost import XGBoostRegressorSubModel
from models.sub_models.m5p import M5PSubModel
from models.fps_model import StackedModel
import random
random.seed(42)

from qpslib_model_manager import model_client
from qpslib_retrain_manager import retrain_client
from common.error import make_error_msg
from common.handler import Handler
from models.preprocess import features
from models.conn_obj import DBConn
from models.conf import ModelParams

from config import Config
import config

#

class FPSModelUpdate(Handler):
    def __init__(self):
        cfg = Config()
        self._service_host = cfg.get_service_host()
        self._db_info = cfg.get_db_info()
        self.local_tz = cfg.get_timezone()
        self.summ_date_name = ModelParams.local_date
        self.train_days = ModelParams.train_days
        self.use_params = ModelParams.total_pow_params
        self.stacked_model = None
        self.outlier = ModelParams.outlier

    # @concurrent.process(timeout=30) # not work with
    def run(self, data: SimpleNamespace):
        # Data setting
        company = data.company
        target = data.target
        master_id = data.master_id
        result_type = data.result_type
        print(f'Start Update : {master_id}')

        output = {}
        comment = ''
        client = model_client.FPSModelClient(self._service_host, company, target, result_type)
        rt_client = retrain_client.FPSRetrainClient(self._service_host, company, target, master_id, result_type)

        # Load model
        try:
            model_info = client.get_best_model()
            print(f'model_id: {model_info.id}')
            self.stacked_model = dill.loads(client.download_model(model_info.id))
        except Exception as e:
            output['error'] = make_error_msg(str(e), f'Load Model: {traceback.format_exc()}')
            print(f'output: {output}')
            return output

        # history 데이터에 대한 summary 생성 및 저장

        # Load Summary data
        try:
            train_end = datetime.strptime(master_id.split('_')[0], '%Y%m%d%H')
            train_start = train_end - timedelta(days=self.train_days)
            train_start_str = datetime.strftime(train_start, '%Y-%m-%d %H:%M:%S')
            train_end_str = datetime.strftime(train_end, '%Y-%m-%d %H:%M:%S')
            collection = DBConn().esbr_summ_db[target]
            df = features.load_features_with_duration(collection, self.summ_date_name, train_start_str, train_end_str)
        except Exception as e:
            output['error'] = make_error_msg(str(e), f'Load Summary data: {traceback.format_exc()}')
            return output

        # Split data
        try:
            X, y = self.split_data(df)
            X = X.fillna(0)
        except Exception as e:
            output['error'] = make_error_msg(str(e), f'Create retrain data: {traceback.format_exc()}')
            return output

        if not rt_client.is_retraining():
            try:
                # Sub-model train
                rt_client.start_retrain()
                sub = M5PSubModel()
                sub_yhat = sub.fit_predict(X, y)
                sub.set_feature_importance(X=X)

                # Stacked-model
                sub_models = {
                    sub.name: sub,
                }
                stacked_model = StackedModel(meta_model=sub,
                                             sub_models=sub_models,
                                             preprocessing=None,
                                             features=self.use_params
                                             )
                # # Sub-model train
                # rt_client.start_retrain()
                # xgb_sub = XGBoostRegressorSubModel()
                # xgb_yhat = xgb_sub.fit_predict(X, y)
                # xgb_sub.set_feature_importance(X=X)
                #
                # # Stacked-model
                # sub_models = {
                #     xgb_sub.name: xgb_sub,
                # }
                # stacked_model = StackedModel(meta_model=xgb_sub,
                #                              sub_models=sub_models,
                #                              preprocessing=None,
                #                              features=self.use_params
                #                              )
            except Exception as e:
                output['error'] = make_error_msg(str(e), f'Retrain: {traceback.format_exc()}')
                return output
            finally:
                rt_client.finish_retrain()

            # Save model
            try:
                stacked_id, sub_id = self.save_model(company, target, client, sub_models, stacked_model)
            except Exception as e:
                output['error'] = make_error_msg(str(e), f'Save model: {traceback.format_exc()}')
                return output

            # output
            output = {'new_model': stacked_id,
                      'evaluation': {'used_data': [df['master_id'][0], df['master_id'][df.index[-1]]],
                                     'models': [{'sub_model': sub_id,
                                                 'stacked_model': stacked_id,
                                                 'rmse': sub.scores['rmse'],
                                                 'R2': sub.scores['r2'],
                                                 '100-mape': sub.scores['100-mape'],
                                                 '100-smape': sub.scores['100-smape']
                                                 }],
                                     'best_model': stacked_id},
                      'retrain': {'sub_model': sub_id,
                                  'stacked_model': stacked_id}}
        else:
            comment += 'Previous retrain is not finished.'
            output = {'new_model': model_info.id,
                        'comment': comment}

        return output


    def split_data(self, data):
        X = data.loc[:, self.use_params].astype(float)
        y = data['y']
        return X, y


    def save_model(self, company, target, client, sub_models, stacked_model):
        sub_models_name = f'Submodels : {len(sub_models.keys())}'
        stacked_model_name = 'EIP Test'

        # Save
        sub_save_resp = model_save.save_sub_models(client=client,
                                                   sub_models=sub_models,
                                                   name=sub_models_name)

        stacked_save_resp = model_save.save_stacked_model(client=client,
                                                          stacked_model=stacked_model,
                                                          ref_id=sub_save_resp.id,
                                                          score=stacked_model.meta_model.scores,
                                                          name=stacked_model_name,
                                                          feats=stacked_model.meta_model.feature_importance)
        client.set_best_model(stacked_save_resp.id)
        return stacked_save_resp.id, sub_save_resp.id



if __name__ == '__main__':
    esbr_input = '''
    {"company":"EIP","target":"MVP_ERCOT_HOUSTON","service_type":"FPS","input_case":"MODEL_UPDATE","result_type":"REGRESSION","master_id":"2022020200_CST","residual":0}
    '''

    start = timeit.default_timer()
    config.load_config('../../config.yml')

    data = SimpleNamespace(**json.loads(esbr_input))

    runner = FPSModelUpdate()
    output = runner.run(data)
    print('Elapse : ', timeit.default_timer() - start)
    print(output)

