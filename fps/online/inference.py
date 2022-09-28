from types import SimpleNamespace
import pandas as pd
import numpy as np
from pytz import utc
import dateutil
import dill
import json
import grpc
import shap
import traceback
from datetime import datetime

from qpslib_model_manager import model_client
from common.error import make_error_msg
from common.handler import Handler
from models.conn_obj import DBConn
from models.preprocess import features
from models.preprocess.preprocessing_y import CreateDate, SummaryData
from models.conf import ModelParams

from config import Config
import config

import warnings
warnings.filterwarnings( 'ignore' )



#%%

class FPSInference(Handler):
    def __init__(self):
        cfg = Config()
        self._service_host = cfg.get_service_host()
        self.local_tz = cfg.get_timezone()
        self.stacked_model = None
        self.summ_date_name = ModelParams.local_date
        self.x_params_to_save = ModelParams.day_ahead_params + ModelParams.anci_params

    # @concurrent.process(timeout=30) # not work with
    def run(self, data: SimpleNamespace):
        # setting
        company = data.company
        target = data.target
        master_id = data.master_id
        predict_time = utc.localize(datetime.strptime(data.time, '%Y-%m-%dT%H:%M:%SZ')).astimezone(self.local_tz)
        result_type = data.result_type
        client = model_client.FPSModelClient(self._service_host, company, target, result_type)
        print(f'Start Inference : {master_id}')
        output = {}

        # Load model
        try:
            model_info = client.get_best_model()
            print(f'model_id: {model_info.id}')
            output['model_id'] = model_info.id
            self.stacked_model = dill.loads(client.download_model(model_info.id))
            # preps = self.stacked_model.preprocessing
            use_feats = self.stacked_model.features
        except grpc.RpcError as e:
            output['error'] = make_error_msg(str(e), f'Load Model: {traceback.format_exc()}')
            print(f'output: {output}')
            return output

        # # Load test model
        # with open('../../data/stacked.pkl', 'rb') as f:
        #     self.stacked_model = dill.load(f)
        # preps = self.stacked_model.preprocessing

        # Create dataset for Preprocessing
        try:

            forecast_date = datetime.strftime(predict_time.date(), '%Y-%m-%d')
            x_df = pd.DataFrame(data.x)

            x_df[self.summ_date_name] = x_df['time'].apply(lambda x: dateutil.parser.parse(x).astimezone(self.local_tz))
            week_list = x_df[self.summ_date_name].dt.day_name().unique().tolist()
            x_df[self.summ_date_name] = x_df[self.summ_date_name].apply(lambda x: datetime.strftime(x, '%Y-%m-%d %H:%M:%S'))

            history_df = x_df.loc[x_df[self.summ_date_name] < forecast_date, :]
            history_df = history_df.copy()
            history_df['y'] = history_df['values'].apply(lambda x: dict(x)['price'])
            history_df = history_df.loc[:, ['time', self.summ_date_name, 'y']]
            history_df.columns = history_df.columns.str.replace('time', 'date')

            forecast_raw = x_df.loc[x_df[self.summ_date_name] >= forecast_date, :]
            forecast_df = pd.DataFrame(list(forecast_raw['values']))
            forecast_df.insert(0, self.summ_date_name, forecast_raw[self.summ_date_name].values)
        except Exception as e:
            output['error'] = make_error_msg(str(e), f'Create feature datasets: {traceback.format_exc()}')
            return output

        # Preprocessing y
        try:
            date_obj = CreateDate('date', self.local_tz)
            history_temp = date_obj.transform(history_df, tag='inference')
            summ_obj = SummaryData('y', self.summ_date_name)
            dummy_val_list = ['week_'+x for x in week_list]
            summ_obj.set_dummy_val_list(dummy_val_list)
            history_preprocessed = summ_obj.transform(history_temp, use_summ=True, tag='inference', inference_date=predict_time)
            hist_feats = history_preprocessed.loc[history_preprocessed[self.summ_date_name].dt.hour == predict_time.hour, :]
            hist_feats = hist_feats.drop([self.summ_date_name], axis=1)

            hist_feats_pow_cols = ['D_1', 'D_2', 'D_3', 'D_4', 'D_5', 'D_6', 'D_7',
                                   'day_before_min', 'day_before_max', 'day_before_avg']
            hist_feats = pd.concat([hist_feats, self.create_pow_feats(hist_feats.loc[:, hist_feats_pow_cols])], axis=1).reset_index(drop=True)
        except Exception as e:
            output['error'] = make_error_msg(str(e), f'Preprocess y for inference: {traceback.format_exc()}')
            return output

        # Preprocessing X
        try:
            forecast_df[self.summ_date_name] = forecast_df[self.summ_date_name].apply(lambda x: dateutil.parser.parse(x))
            fore_feats = forecast_df.loc[forecast_df[self.summ_date_name].dt.hour == predict_time.hour, :]
            fore_feats[self.summ_date_name] = fore_feats[self.summ_date_name].apply(lambda x: datetime.strftime(x, '%Y-%m-%d %H:%M:%S'))

            fore_feats_pow_temp = fore_feats.drop(self.summ_date_name, axis=1)
            fore_feats = pd.concat([fore_feats, self.create_pow_feats(fore_feats_pow_temp)], axis=1).reset_index(drop=True)
        except Exception as e:
            output['error'] = make_error_msg(str(e), f'Preprocess X for inference: {traceback.format_exc()}')
            return output

        # Merge and fill NA
        try:
            summary_df = pd.concat([hist_feats, fore_feats], axis=1)
            summary_df.insert(0, 'master_id', master_id)
            no_value_params = [p for p in use_feats if p not in summary_df.columns]
            summary_df[no_value_params] = np.nan
        except Exception as e:
            output['error'] = make_error_msg(str(e), f'Merge y and x to make inference data: {traceback.format_exc()}')
            return output


        # Save inference data to DB
        try:
            collection = DBConn().esbr_summ_db[target]
            features_to_save = summary_df.drop('y', axis=1)
            features.save_features_to_db(collection, features_to_save)
        except Exception as e:
            output['error'] = make_error_msg(str(e), f'Save inference data to DB: {traceback.format_exc()}')
            return output

        # Predict
        try:
            # inference_df = summary_df.drop(['master_id', self.summ_date_name, 'week'], axis=1)
            inference_df = summary_df.loc[:, use_feats].astype(float)
            inference_df = inference_df.fillna(0)

            y_hat = float(self.stacked_model.meta_model.model.predict(inference_df)[0])
            y_hat = 0 if y_hat < 0 else y_hat
            y_hat = (ModelParams.outlier + np.sqrt(y_hat)) if y_hat > ModelParams.outlier else y_hat
            # pred = np.array([0 if x<0 else x for x in pred ])

        except Exception as e:
            output['error'] = make_error_msg(str(e), f'Predict : {traceback.format_exc()}')
            return output

        # contribution
        try:
            # contribution = self.get_contribution(inference_df)
            contribution = {'temp':0}
        except Exception as e:
            output['error'] = make_error_msg(str(e), f'Getting contribution : {traceback.format_exc()}')
            return output

        output['y_hat'] = y_hat
        output['contribution'] = contribution

        return output


    def get_contribution(self, X):
        contribution_limit = 1e-2
        meta_model = self.stacked_model.meta_model
        explainer = shap.TreeExplainer(meta_model.model)
        shap_values = explainer.shap_values(X.values)
        ctb_raw = pd.Series(data=shap_values[0],
                            index=X.columns)
        contribution = ctb_raw.abs().round(3)
        contribution = dict(contribution[contribution > contribution_limit])
        contribution = {k: float(v/sum(list(contribution.values()))) for k, v in contribution.items()}

        return contribution

    def create_pow_feats(self, data):
        pow_data = data.apply(lambda x: x ** 2)
        pow_data.columns = [x + '_pow' for x in data.columns]
        return pow_data


if __name__ == '__main__':

  #   esbr_input = '''
  # {"company": "EIP", "target": "MVP_HOUSTON", "service_type": "FPS", "input_case": "INFERENCE", "result_type": "REGRESSION", "master_id": "2022011500_CST", "time": "2022-01-15T06:00:00Z", "x": [{"time": "2022-01-08T06:00:00Z", "values": {"price": 28.7525}}, {"time": "2022-01-08T07:00:00Z", "values": {"price": 27.455}}, {"time": "2022-01-08T08:00:00Z", "values": {"price": 26.4625}}, {"time": "2022-01-08T09:00:00Z", "values": {"price": 26.1575}}, {"time": "2022-01-08T10:00:00Z", "values": {"price": 25.75}}, {"time": "2022-01-08T11:00:00Z", "values": {"price": 25.0875}}, {"time": "2022-01-08T12:00:00Z", "values": {"price": 23.1425}}, {"time": "2022-01-08T13:00:00Z", "values": {"price": 23.0275}}, {"time": "2022-01-08T14:00:00Z", "values": {"price": 24.9025}}, {"time": "2022-01-08T15:00:00Z", "values": {"price": 26.2375}}, {"time": "2022-01-08T16:00:00Z", "values": {"price": 25.165}}, {"time": "2022-01-08T17:00:00Z", "values": {"price": 26.4475}}, {"time": "2022-01-08T18:00:00Z", "values": {"price": 26.585}}, {"time": "2022-01-08T19:00:00Z", "values": {"price": 26.02}}, {"time": "2022-01-08T20:00:00Z", "values": {"price": 24.04}}, {"time": "2022-01-08T21:00:00Z", "values": {"price": 22.4075}}, {"time": "2022-01-08T22:00:00Z", "values": {"price": 23.11}}, {"time": "2022-01-08T23:00:00Z", "values": {"price": 132.0725}}, {"time": "2022-01-09T00:00:00Z", "values": {"price": 90.465}}, {"time": "2022-01-09T01:00:00Z", "values": {"price": 27.5525}}, {"time": "2022-01-09T02:00:00Z", "values": {"price": 24.345}}, {"time": "2022-01-09T03:00:00Z", "values": {"price": 21.5575}}, {"time": "2022-01-09T04:00:00Z", "values": {"price": 22.0275}}, {"time": "2022-01-09T05:00:00Z", "values": {"price": 21.505}}, {"time": "2022-01-09T06:00:00Z", "values": {"price": 22.535}}, {"time": "2022-01-09T07:00:00Z", "values": {"price": 19.8175}}, {"time": "2022-01-09T08:00:00Z", "values": {"price": 4.77}}, {"time": "2022-01-09T09:00:00Z", "values": {"price": 1.9425}}, {"time": "2022-01-09T10:00:00Z", "values": {"price": -0.045}}, {"time": "2022-01-09T11:00:00Z", "values": {"price": 6.185}}, {"time": "2022-01-09T12:00:00Z", "values": {"price": 16.985}}, {"time": "2022-01-09T13:00:00Z", "values": {"price": 17.05}}, {"time": "2022-01-09T14:00:00Z", "values": {"price": 18.3525}}, {"time": "2022-01-09T15:00:00Z", "values": {"price": 20.9375}}, {"time": "2022-01-09T16:00:00Z", "values": {"price": 21.2325}}, {"time": "2022-01-09T17:00:00Z", "values": {"price": 21.69}}, {"time": "2022-01-09T18:00:00Z", "values": {"price": 22.3325}}, {"time": "2022-01-09T19:00:00Z", "values": {"price": 23.055}}, {"time": "2022-01-09T20:00:00Z", "values": {"price": 23.9575}}, {"time": "2022-01-09T21:00:00Z", "values": {"price": 19.705}}, {"time": "2022-01-09T22:00:00Z", "values": {"price": 20.4625}}, {"time": "2022-01-09T23:00:00Z", "values": {"price": 25.1975}}, {"time": "2022-01-10T00:00:00Z", "values": {"price": 27.4125}}, {"time": "2022-01-10T01:00:00Z", "values": {"price": 25.575}}, {"time": "2022-01-10T02:00:00Z", "values": {"price": 25.4875}}, {"time": "2022-01-10T03:00:00Z", "values": {"price": 22.8825}}, {"time": "2022-01-10T04:00:00Z", "values": {"price": 21.84}}, {"time": "2022-01-10T05:00:00Z", "values": {"price": 22.1}}, {"time": "2022-01-10T06:00:00Z", "values": {"price": 21.595}}, {"time": "2022-01-10T07:00:00Z", "values": {"price": 21.1325}}, {"time": "2022-01-10T08:00:00Z", "values": {"price": 21.7225}}, {"time": "2022-01-10T09:00:00Z", "values": {"price": 21.675}}, {"time": "2022-01-10T10:00:00Z", "values": {"price": 21.46}}, {"time": "2022-01-10T11:00:00Z", "values": {"price": 23.0375}}, {"time": "2022-01-10T12:00:00Z", "values": {"price": 38.985}}, {"time": "2022-01-10T13:00:00Z", "values": {"price": 70.24}}, {"time": "2022-01-10T14:00:00Z", "values": {"price": 61.7}}, {"time": "2022-01-10T15:00:00Z", "values": {"price": 37.475}}, {"time": "2022-01-10T16:00:00Z", "values": {"price": 36.245}}, {"time": "2022-01-10T17:00:00Z", "values": {"price": 27.6675}}, {"time": "2022-01-10T18:00:00Z", "values": {"price": 25.5325}}, {"time": "2022-01-10T19:00:00Z", "values": {"price": 23.525}}, {"time": "2022-01-10T20:00:00Z", "values": {"price": 22.54}}, {"time": "2022-01-10T21:00:00Z", "values": {"price": 23.185}}, {"time": "2022-01-10T22:00:00Z", "values": {"price": 27.3125}}, {"time": "2022-01-10T23:00:00Z", "values": {"price": 49.27}}, {"time": "2022-01-11T00:00:00Z", "values": {"price": 40.6425}}, {"time": "2022-01-11T01:00:00Z", "values": {"price": 37.6275}}, {"time": "2022-01-11T02:00:00Z", "values": {"price": 38.1275}}, {"time": "2022-01-11T03:00:00Z", "values": {"price": 37.3825}}, {"time": "2022-01-11T04:00:00Z", "values": {"price": 33.095}}, {"time": "2022-01-11T05:00:00Z", "values": {"price": 28.7975}}, {"time": "2022-01-11T06:00:00Z", "values": {"price": 27.8675}}, {"time": "2022-01-11T07:00:00Z", "values": {"price": 25.5825}}, {"time": "2022-01-11T08:00:00Z", "values": {"price": 24.9525}}, {"time": "2022-01-11T09:00:00Z", "values": {"price": 25.125}}, {"time": "2022-01-11T10:00:00Z", "values": {"price": 25.9675}}, {"time": "2022-01-11T11:00:00Z", "values": {"price": 28.515}}, {"time": "2022-01-11T12:00:00Z", "values": {"price": 36.8175}}, {"time": "2022-01-11T13:00:00Z", "values": {"price": 40.3525}}, {"time": "2022-01-11T14:00:00Z", "values": {"price": 36.255}}, {"time": "2022-01-11T15:00:00Z", "values": {"price": 39.905}}, {"time": "2022-01-11T16:00:00Z", "values": {"price": 28.01}}, {"time": "2022-01-11T17:00:00Z", "values": {"price": 23.7075}}, {"time": "2022-01-11T18:00:00Z", "values": {"price": 22.345}}, {"time": "2022-01-11T19:00:00Z", "values": {"price": 20.415}}, {"time": "2022-01-11T20:00:00Z", "values": {"price": 18.5625}}, {"time": "2022-01-11T21:00:00Z", "values": {"price": 20.18}}, {"time": "2022-01-11T22:00:00Z", "values": {"price": 26.7725}}, {"time": "2022-01-11T23:00:00Z", "values": {"price": 75.2175}}, {"time": "2022-01-12T00:00:00Z", "values": {"price": 35.66}}, {"time": "2022-01-12T01:00:00Z", "values": {"price": 29.7375}}, {"time": "2022-01-12T02:00:00Z", "values": {"price": 26.66}}, {"time": "2022-01-12T03:00:00Z", "values": {"price": 24.9275}}, {"time": "2022-01-12T04:00:00Z", "values": {"price": 22.725}}, {"time": "2022-01-12T05:00:00Z", "values": {"price": 21.8875}}, {"time": "2022-01-12T06:00:00Z", "values": {"price": 21.9375}}, {"time": "2022-01-12T07:00:00Z", "values": {"price": 21.98}}, {"time": "2022-01-12T08:00:00Z", "values": {"price": 22.4475}}, {"time": "2022-01-12T09:00:00Z", "values": {"price": 23.3525}}, {"time": "2022-01-12T10:00:00Z", "values": {"price": 24.0275}}, {"time": "2022-01-12T11:00:00Z", "values": {"price": 26.905}}, {"time": "2022-01-12T12:00:00Z", "values": {"price": 34.0875}}, {"time": "2022-01-12T13:00:00Z", "values": {"price": 38.5425}}, {"time": "2022-01-12T14:00:00Z", "values": {"price": 31.5625}}, {"time": "2022-01-12T15:00:00Z", "values": {"price": 25.8475}}, {"time": "2022-01-12T16:00:00Z", "values": {"price": 27.88}}, {"time": "2022-01-12T17:00:00Z", "values": {"price": 26.22}}, {"time": "2022-01-12T18:00:00Z", "values": {"price": 25.1725}}, {"time": "2022-01-12T19:00:00Z", "values": {"price": 23.3575}}, {"time": "2022-01-12T20:00:00Z", "values": {"price": 22.45}}, {"time": "2022-01-12T21:00:00Z", "values": {"price": 22.7325}}, {"time": "2022-01-12T22:00:00Z", "values": {"price": 23.605}}, {"time": "2022-01-12T23:00:00Z", "values": {"price": 32.48}}, {"time": "2022-01-13T00:00:00Z", "values": {"price": 39.175}}, {"time": "2022-01-13T01:00:00Z", "values": {"price": 43.0775}}, {"time": "2022-01-13T02:00:00Z", "values": {"price": 38.1325}}, {"time": "2022-01-13T03:00:00Z", "values": {"price": 36.3725}}, {"time": "2022-01-13T04:00:00Z", "values": {"price": 33.0525}}, {"time": "2022-01-13T05:00:00Z", "values": {"price": 36.515}}, {"time": "2022-01-13T06:00:00Z", "values": {"price": 34.7325}}, {"time": "2022-01-13T07:00:00Z", "values": {"price": 31.0475}}, {"time": "2022-01-13T08:00:00Z", "values": {"price": 35.6025}}, {"time": "2022-01-13T09:00:00Z", "values": {"price": 36.7075}}, {"time": "2022-01-13T10:00:00Z", "values": {"price": 39.655}}, {"time": "2022-01-13T11:00:00Z", "values": {"price": 44.755}}, {"time": "2022-01-13T12:00:00Z", "values": {"price": 62.535}}, {"time": "2022-01-13T13:00:00Z", "values": {"price": 101.43}}, {"time": "2022-01-13T14:00:00Z", "values": {"price": 34.4425}}, {"time": "2022-01-13T15:00:00Z", "values": {"price": 28.685}}, {"time": "2022-01-13T16:00:00Z", "values": {"price": 30.5775}}, {"time": "2022-01-13T17:00:00Z", "values": {"price": 27.675}}, {"time": "2022-01-13T18:00:00Z", "values": {"price": 26.6325}}, {"time": "2022-01-13T19:00:00Z", "values": {"price": 26.265}}, {"time": "2022-01-13T20:00:00Z", "values": {"price": 25.72}}, {"time": "2022-01-13T21:00:00Z", "values": {"price": 25.9075}}, {"time": "2022-01-13T22:00:00Z", "values": {"price": 28.1275}}, {"time": "2022-01-13T23:00:00Z", "values": {"price": 42.3825}}, {"time": "2022-01-14T00:00:00Z", "values": {"price": 52.175}}, {"time": "2022-01-14T01:00:00Z", "values": {"price": 47.475}}, {"time": "2022-01-14T02:00:00Z", "values": {"price": 46.2425}}, {"time": "2022-01-14T03:00:00Z", "values": {"price": 36.3875}}, {"time": "2022-01-14T04:00:00Z", "values": {"price": 29.5075}}, {"time": "2022-01-14T05:00:00Z", "values": {"price": 29.54}}, {"time": "2022-01-14T06:00:00Z", "values": {"price": 27.0875}}, {"time": "2022-01-14T07:00:00Z", "values": {"price": 23.92}}, {"time": "2022-01-14T08:00:00Z", "values": {"price": 23.3275}}, {"time": "2022-01-14T09:00:00Z", "values": {"price": 24.86}}, {"time": "2022-01-14T10:00:00Z", "values": {"price": 24.4075}}, {"time": "2022-01-14T11:00:00Z", "values": {"price": 25.065}}, {"time": "2022-01-14T12:00:00Z", "values": {"price": 29.9425}}, {"time": "2022-01-14T13:00:00Z", "values": {"price": 30.495}}, {"time": "2022-01-14T14:00:00Z", "values": {"price": 29.625}}, {"time": "2022-01-14T15:00:00Z", "values": {"price": 29.29}}, {"time": "2022-01-14T16:00:00Z", "values": {"price": 25.695}}, {"time": "2022-01-14T17:00:00Z", "values": {"price": 23.22}}, {"time": "2022-01-14T18:00:00Z", "values": {"price": 22.8425}}, {"time": "2022-01-14T19:00:00Z", "values": {"price": 23.12}}, {"time": "2022-01-14T20:00:00Z", "values": {"price": 23.52}}, {"time": "2022-01-14T21:00:00Z", "values": {"price": 24.5}}, {"time": "2022-01-14T22:00:00Z", "values": {"price": 23.205}}, {"time": "2022-01-14T23:00:00Z", "values": {"price": 27.6825}}, {"time": "2022-01-15T00:00:00Z", "values": {"price": 23.6875}}, {"time": "2022-01-15T01:00:00Z", "values": {"price": 17.9275}}, {"time": "2022-01-15T02:00:00Z", "values": {"price": 28.38}}, {"time": "2022-01-15T03:00:00Z", "values": {"price": 19.7125}}, {"time": "2022-01-15T04:00:00Z", "values": {"price": 16.6075}}, {"time": "2022-01-15T05:00:00Z", "values": {"price": 9.0775}}, {"time": "2022-01-15T06:00:00Z", "values": {"day_ahead_price": 13.25, "mcpc_non_spin": 0.96, "mcpc_reg_down": 4.25, "mcpc_reg_up": 4.25, "mcpc_rrs": 4.15}}, {"time": "2022-01-15T07:00:00Z", "values": {"day_ahead_price": 13.54, "mcpc_non_spin": 0.96, "mcpc_reg_down": 4.0, "mcpc_reg_up": 4.25, "mcpc_rrs": 4.15}}, {"time": "2022-01-15T08:00:00Z", "values": {"day_ahead_price": 14.0, "mcpc_non_spin": 0.97, "mcpc_reg_down": 4.0, "mcpc_reg_up": 4.25, "mcpc_rrs": 4.15}}, {"time": "2022-01-15T09:00:00Z", "values": {"day_ahead_price": 14.8, "mcpc_non_spin": 0.97, "mcpc_reg_down": 5.5, "mcpc_reg_up": 5.5, "mcpc_rrs": 4.9}}, {"time": "2022-01-15T10:00:00Z", "values": {"day_ahead_price": 17.48, "mcpc_non_spin": 0.97, "mcpc_reg_down": 6.41, "mcpc_reg_up": 6.01, "mcpc_rrs": 5.91}}, {"time": "2022-01-15T11:00:00Z", "values": {"day_ahead_price": 17.48, "mcpc_non_spin": 1.0, "mcpc_reg_down": 10.7, "mcpc_reg_up": 10.7, "mcpc_rrs": 10.1}}, {"time": "2022-01-15T12:00:00Z", "values": {"day_ahead_price": 29.3, "mcpc_non_spin": 2.85, "mcpc_reg_down": 12.0, "mcpc_reg_up": 17.37, "mcpc_rrs": 16.57}}, {"time": "2022-01-15T13:00:00Z", "values": {"day_ahead_price": 26.98, "mcpc_non_spin": 3.01, "mcpc_reg_down": 10.95, "mcpc_reg_up": 10.85, "mcpc_rrs": 10.85}}, {"time": "2022-01-15T14:00:00Z", "values": {"day_ahead_price": 28.29, "mcpc_non_spin": 2.85, "mcpc_reg_down": 10.85, "mcpc_reg_up": 10.85, "mcpc_rrs": 10.0}}, {"time": "2022-01-15T15:00:00Z", "values": {"day_ahead_price": 29.72, "mcpc_non_spin": 2.64, "mcpc_reg_down": 10.0, "mcpc_reg_up": 10.0, "mcpc_rrs": 9.0}}, {"time": "2022-01-15T16:00:00Z", "values": {"day_ahead_price": 31.87, "mcpc_non_spin": 1.7, "mcpc_reg_down": 9.5, "mcpc_reg_up": 7.73, "mcpc_rrs": 7.15}}, {"time": "2022-01-15T17:00:00Z", "values": {"day_ahead_price": 32.44, "mcpc_non_spin": 1.7, "mcpc_reg_down": 7.73, "mcpc_reg_up": 7.73, "mcpc_rrs": 6.5}}, {"time": "2022-01-15T18:00:00Z", "values": {"day_ahead_price": 29.47, "mcpc_non_spin": 2.2, "mcpc_reg_down": 8.0, "mcpc_reg_up": 7.0, "mcpc_rrs": 5.15}}, {"time": "2022-01-15T19:00:00Z", "values": {"day_ahead_price": 26.23, "mcpc_non_spin": 2.2, "mcpc_reg_down": 8.0, "mcpc_reg_up": 6.15, "mcpc_rrs": 4.01}}, {"time": "2022-01-15T20:00:00Z", "values": {"day_ahead_price": 25.96, "mcpc_non_spin": 3.14, "mcpc_reg_down": 8.83, "mcpc_reg_up": 6.83, "mcpc_rrs": 5.15}}, {"time": "2022-01-15T21:00:00Z", "values": {"day_ahead_price": 25.41, "mcpc_non_spin": 3.14, "mcpc_reg_down": 8.0, "mcpc_reg_up": 8.0, "mcpc_rrs": 6.37}}, {"time": "2022-01-15T22:00:00Z", "values": {"day_ahead_price": 25.65, "mcpc_non_spin": 4.4, "mcpc_reg_down": 8.18, "mcpc_reg_up": 8.18, "mcpc_rrs": 4.01}}, {"time": "2022-01-15T23:00:00Z", "values": {"day_ahead_price": 43.42, "mcpc_non_spin": 12.51, "mcpc_reg_down": 3.5, "mcpc_reg_up": 19.79, "mcpc_rrs": 16.88}}, {"time": "2022-01-16T00:00:00Z", "values": {"day_ahead_price": 43.39, "mcpc_non_spin": 4.62, "mcpc_reg_down": 5.0, "mcpc_reg_up": 9.55, "mcpc_rrs": 9.6}}, {"time": "2022-01-16T01:00:00Z", "values": {"day_ahead_price": 41.84, "mcpc_non_spin": 3.14, "mcpc_reg_down": 5.0, "mcpc_reg_up": 3.14, "mcpc_rrs": 6.5}}, {"time": "2022-01-16T02:00:00Z", "values": {"day_ahead_price": 44.61, "mcpc_non_spin": 3.14, "mcpc_reg_down": 5.0, "mcpc_reg_up": 4.25, "mcpc_rrs": 5.0}}, {"time": "2022-01-16T03:00:00Z", "values": {"day_ahead_price": 46.28, "mcpc_non_spin": 3.08, "mcpc_reg_down": 4.0, "mcpc_reg_up": 5.0, "mcpc_rrs": 5.0}}, {"time": "2022-01-16T04:00:00Z", "values": {"day_ahead_price": 43.86, "mcpc_non_spin": 1.35, "mcpc_reg_down": 4.0, "mcpc_reg_up": 4.5, "mcpc_rrs": 4.5}}, {"time": "2022-01-16T05:00:00Z", "values": {"day_ahead_price": 41.72, "mcpc_non_spin": 1.35, "mcpc_reg_down": 4.0, "mcpc_reg_up": 4.56, "mcpc_rrs": 4.46}}]}
  #   '''
    esbr_input = '''
    {"company":"EIP","target":"MVP_ERCOT_HOUSTON","service_type":"FPS","input_case":"INFERENCE","result_type":"REGRESSION","master_id":"2022040100_CDT","time":"2022-04-01T05:00:00Z","x":[{"time":"2022-03-24T05:00:00Z","values":{"price":52.7825}},{"time":"2022-03-24T06:00:00Z","values":{"price":58.38}},{"time":"2022-03-24T07:00:00Z","values":{"price":33.1625}},{"time":"2022-03-24T08:00:00Z","values":{"price":31.795}},{"time":"2022-03-24T09:00:00Z","values":{"price":28.6125}},{"time":"2022-03-24T10:00:00Z","values":{"price":32.0375}},{"time":"2022-03-24T11:00:00Z","values":{"price":46.725}},{"time":"2022-03-24T12:00:00Z","values":{"price":50.3475}},{"time":"2022-03-24T13:00:00Z","values":{"price":22.53}},{"time":"2022-03-24T14:00:00Z","values":{"price":37.4025}},{"time":"2022-03-24T15:00:00Z","values":{"price":32.7575}},{"time":"2022-03-24T16:00:00Z","values":{"price":21.7725}},{"time":"2022-03-24T17:00:00Z","values":{"price":12.74}},{"time":"2022-03-24T18:00:00Z","values":{"price":23.66}},{"time":"2022-03-24T19:00:00Z","values":{"price":25.1675}},{"time":"2022-03-24T20:00:00Z","values":{"price":24.65}},{"time":"2022-03-24T21:00:00Z","values":{"price":20.9725}},{"time":"2022-03-24T22:00:00Z","values":{"price":17.41}},{"time":"2022-03-24T23:00:00Z","values":{"price":26.94}},{"time":"2022-03-25T00:00:00Z","values":{"price":95.0125}},{"time":"2022-03-25T01:00:00Z","values":{"price":84.415}},{"time":"2022-03-25T02:00:00Z","values":{"price":63.3825}},{"time":"2022-03-25T03:00:00Z","values":{"price":47.49}},{"time":"2022-03-25T04:00:00Z","values":{"price":31.965}},{"time":"2022-03-25T05:00:00Z","values":{"price":29.7875}},{"time":"2022-03-25T06:00:00Z","values":{"price":28.0725}},{"time":"2022-03-25T07:00:00Z","values":{"price":26.9425}},{"time":"2022-03-25T08:00:00Z","values":{"price":27.48}},{"time":"2022-03-25T09:00:00Z","values":{"price":29.2375}},{"time":"2022-03-25T10:00:00Z","values":{"price":35.4025}},{"time":"2022-03-25T11:00:00Z","values":{"price":50.34}},{"time":"2022-03-25T12:00:00Z","values":{"price":63.52}},{"time":"2022-03-25T13:00:00Z","values":{"price":42.745}},{"time":"2022-03-25T14:00:00Z","values":{"price":37.71}},{"time":"2022-03-25T15:00:00Z","values":{"price":37.0225}},{"time":"2022-03-25T16:00:00Z","values":{"price":34.045}},{"time":"2022-03-25T17:00:00Z","values":{"price":33.74}},{"time":"2022-03-25T18:00:00Z","values":{"price":42.57}},{"time":"2022-03-25T19:00:00Z","values":{"price":41.3675}},{"time":"2022-03-25T20:00:00Z","values":{"price":57.9675}},{"time":"2022-03-25T21:00:00Z","values":{"price":45.2825}},{"time":"2022-03-25T22:00:00Z","values":{"price":42.6725}},{"time":"2022-03-25T23:00:00Z","values":{"price":45.535}},{"time":"2022-03-26T00:00:00Z","values":{"price":59.25}},{"time":"2022-03-26T01:00:00Z","values":{"price":40.17}},{"time":"2022-03-26T02:00:00Z","values":{"price":33.375}},{"time":"2022-03-26T03:00:00Z","values":{"price":31.77}},{"time":"2022-03-26T04:00:00Z","values":{"price":19.8425}},{"time":"2022-03-26T05:00:00Z","values":{"price":23.0125}},{"time":"2022-03-26T06:00:00Z","values":{"price":26.48}},{"time":"2022-03-26T07:00:00Z","values":{"price":27.1475}},{"time":"2022-03-26T08:00:00Z","values":{"price":40.69}},{"time":"2022-03-26T09:00:00Z","values":{"price":48.855}},{"time":"2022-03-26T10:00:00Z","values":{"price":32.34}},{"time":"2022-03-26T11:00:00Z","values":{"price":33.165}},{"time":"2022-03-26T12:00:00Z","values":{"price":62.5625}},{"time":"2022-03-26T13:00:00Z","values":{"price":46.6725}},{"time":"2022-03-26T14:00:00Z","values":{"price":38.0725}},{"time":"2022-03-26T15:00:00Z","values":{"price":49.145}},{"time":"2022-03-26T16:00:00Z","values":{"price":36.3675}},{"time":"2022-03-26T17:00:00Z","values":{"price":37.4475}},{"time":"2022-03-26T18:00:00Z","values":{"price":36.645}},{"time":"2022-03-26T19:00:00Z","values":{"price":65.405}},{"time":"2022-03-26T20:00:00Z","values":{"price":87.9825}},{"time":"2022-03-26T21:00:00Z","values":{"price":48.075}},{"time":"2022-03-26T22:00:00Z","values":{"price":55.8}},{"time":"2022-03-26T23:00:00Z","values":{"price":131.9675}},{"time":"2022-03-27T00:00:00Z","values":{"price":438.6675}},{"time":"2022-03-27T01:00:00Z","values":{"price":52.93}},{"time":"2022-03-27T02:00:00Z","values":{"price":37.875}},{"time":"2022-03-27T03:00:00Z","values":{"price":61.6375}},{"time":"2022-03-27T04:00:00Z","values":{"price":30.8}},{"time":"2022-03-27T05:00:00Z","values":{"price":32.965}},{"time":"2022-03-27T06:00:00Z","values":{"price":33.085}},{"time":"2022-03-27T07:00:00Z","values":{"price":26.5}},{"time":"2022-03-27T08:00:00Z","values":{"price":27.0575}},{"time":"2022-03-27T09:00:00Z","values":{"price":23.2775}},{"time":"2022-03-27T10:00:00Z","values":{"price":26.2875}},{"time":"2022-03-27T11:00:00Z","values":{"price":30.3525}},{"time":"2022-03-27T12:00:00Z","values":{"price":32.77}},{"time":"2022-03-27T13:00:00Z","values":{"price":29.1475}},{"time":"2022-03-27T14:00:00Z","values":{"price":30.6075}},{"time":"2022-03-27T15:00:00Z","values":{"price":34.72}},{"time":"2022-03-27T16:00:00Z","values":{"price":38.0575}},{"time":"2022-03-27T17:00:00Z","values":{"price":40.065}},{"time":"2022-03-27T18:00:00Z","values":{"price":47.5625}},{"time":"2022-03-27T19:00:00Z","values":{"price":53.5475}},{"time":"2022-03-27T20:00:00Z","values":{"price":50.005}},{"time":"2022-03-27T21:00:00Z","values":{"price":55.1425}},{"time":"2022-03-27T22:00:00Z","values":{"price":47.15}},{"time":"2022-03-27T23:00:00Z","values":{"price":41.8625}},{"time":"2022-03-28T00:00:00Z","values":{"price":48.745}},{"time":"2022-03-28T01:00:00Z","values":{"price":36.7125}},{"time":"2022-03-28T02:00:00Z","values":{"price":44.005}},{"time":"2022-03-28T03:00:00Z","values":{"price":45.1525}},{"time":"2022-03-28T04:00:00Z","values":{"price":28.875}},{"time":"2022-03-28T05:00:00Z","values":{"price":25.405}},{"time":"2022-03-28T06:00:00Z","values":{"price":23.205}},{"time":"2022-03-28T07:00:00Z","values":{"price":22.4}},{"time":"2022-03-28T08:00:00Z","values":{"price":24.015}},{"time":"2022-03-28T09:00:00Z","values":{"price":25.836666}},{"time":"2022-03-28T10:00:00Z","values":{"price":25.365}},{"time":"2022-03-28T11:00:00Z","values":{"price":46.3175}},{"time":"2022-03-28T12:00:00Z","values":{"price":36.855}},{"time":"2022-03-28T13:00:00Z","values":{"price":28.7525}},{"time":"2022-03-28T14:00:00Z","values":{"price":26.155}},{"time":"2022-03-28T15:00:00Z","values":{"price":33.01}},{"time":"2022-03-28T16:00:00Z","values":{"price":35.66}},{"time":"2022-03-28T17:00:00Z","values":{"price":120.255}},{"time":"2022-03-28T18:00:00Z","values":{"price":43.735}},{"time":"2022-03-28T19:00:00Z","values":{"price":46.875}},{"time":"2022-03-28T20:00:00Z","values":{"price":50.645}},{"time":"2022-03-28T21:00:00Z","values":{"price":72.465}},{"time":"2022-03-28T22:00:00Z","values":{"price":52.4675}},{"time":"2022-03-28T23:00:00Z","values":{"price":47.585}},{"time":"2022-03-29T00:00:00Z","values":{"price":52.6725}},{"time":"2022-03-29T01:00:00Z","values":{"price":42.5325}},{"time":"2022-03-29T02:00:00Z","values":{"price":46.365}},{"time":"2022-03-29T03:00:00Z","values":{"price":42.2725}},{"time":"2022-03-29T04:00:00Z","values":{"price":35.0825}},{"time":"2022-03-29T05:00:00Z","values":{"price":25.9325}},{"time":"2022-03-29T06:00:00Z","values":{"price":20.7775}},{"time":"2022-03-29T07:00:00Z","values":{"price":19.155}},{"time":"2022-03-29T08:00:00Z","values":{"price":16.22}},{"time":"2022-03-29T09:00:00Z","values":{"price":19.7975}},{"time":"2022-03-29T10:00:00Z","values":{"price":29.6925}},{"time":"2022-03-29T11:00:00Z","values":{"price":115.5575}},{"time":"2022-03-29T12:00:00Z","values":{"price":65.0975}},{"time":"2022-03-29T13:00:00Z","values":{"price":35.3675}},{"time":"2022-03-29T14:00:00Z","values":{"price":32.715}},{"time":"2022-03-29T15:00:00Z","values":{"price":36.6775}},{"time":"2022-03-29T16:00:00Z","values":{"price":49.04}},{"time":"2022-03-29T17:00:00Z","values":{"price":50.9675}},{"time":"2022-03-29T18:00:00Z","values":{"price":77.5325}},{"time":"2022-03-29T19:00:00Z","values":{"price":96.0225}},{"time":"2022-03-29T20:00:00Z","values":{"price":65.4675}},{"time":"2022-03-29T21:00:00Z","values":{"price":56.2275}},{"time":"2022-03-29T22:00:00Z","values":{"price":44.51}},{"time":"2022-03-29T23:00:00Z","values":{"price":42.32}},{"time":"2022-03-30T00:00:00Z","values":{"price":43.13}},{"time":"2022-03-30T01:00:00Z","values":{"price":47.495}},{"time":"2022-03-30T02:00:00Z","values":{"price":45.6}},{"time":"2022-03-30T03:00:00Z","values":{"price":54.85}},{"time":"2022-03-30T04:00:00Z","values":{"price":87.5725}},{"time":"2022-03-30T05:00:00Z","values":{"price":266.4175}},{"time":"2022-03-30T06:00:00Z","values":{"price":81.98}},{"time":"2022-03-30T07:00:00Z","values":{"price":33.0225}},{"time":"2022-03-30T08:00:00Z","values":{"price":29.015}},{"time":"2022-03-30T09:00:00Z","values":{"price":22.505}},{"time":"2022-03-30T10:00:00Z","values":{"price":36.7225}},{"time":"2022-03-30T11:00:00Z","values":{"price":184.035}},{"time":"2022-03-30T12:00:00Z","values":{"price":49.56}},{"time":"2022-03-30T13:00:00Z","values":{"price":34.3475}},{"time":"2022-03-30T14:00:00Z","values":{"price":36.5275}},{"time":"2022-03-30T15:00:00Z","values":{"price":31.9075}},{"time":"2022-03-30T16:00:00Z","values":{"price":19.865}},{"time":"2022-03-30T17:00:00Z","values":{"price":22.1175}},{"time":"2022-03-30T18:00:00Z","values":{"price":26.8025}},{"time":"2022-03-30T19:00:00Z","values":{"price":29.63}},{"time":"2022-03-30T20:00:00Z","values":{"price":28.955}},{"time":"2022-03-30T21:00:00Z","values":{"price":29.47}},{"time":"2022-03-30T22:00:00Z","values":{"price":32.01}},{"time":"2022-03-30T23:00:00Z","values":{"price":27.6325}},{"time":"2022-03-31T00:00:00Z","values":{"price":46.7625}},{"time":"2022-03-31T01:00:00Z","values":{"price":53.63}},{"time":"2022-03-31T02:00:00Z","values":{"price":39.135}},{"time":"2022-03-31T03:00:00Z","values":{"price":33.4375}},{"time":"2022-03-31T04:00:00Z","values":{"price":29.38}},{"time":"2022-03-31T05:00:00Z","values":{"price":30.935}},{"time":"2022-03-31T06:00:00Z","values":{"price":33.7075}},{"time":"2022-03-31T07:00:00Z","values":{"price":30.92}},{"time":"2022-03-31T08:00:00Z","values":{"price":26.9725}},{"time":"2022-03-31T09:00:00Z","values":{"price":24.7075}},{"time":"2022-03-31T10:00:00Z","values":{"price":29.4025}},{"time":"2022-03-31T11:00:00Z","values":{"price":32.405}},{"time":"2022-03-31T12:00:00Z","values":{"price":37.1025}},{"time":"2022-03-31T13:00:00Z","values":{"price":32.68}},{"time":"2022-03-31T14:00:00Z","values":{"price":31.445}},{"time":"2022-03-31T15:00:00Z","values":{"price":29.1175}},{"time":"2022-03-31T16:00:00Z","values":{"price":27.175}},{"time":"2022-03-31T17:00:00Z","values":{"price":27.9475}},{"time":"2022-03-31T18:00:00Z","values":{"price":28.3825}},{"time":"2022-03-31T19:00:00Z","values":{"price":31.485}},{"time":"2022-03-31T20:00:00Z","values":{"price":33.9575}},{"time":"2022-03-31T21:00:00Z","values":{"price":37.88}},{"time":"2022-03-31T22:00:00Z","values":{"price":43.9025}},{"time":"2022-03-31T23:00:00Z","values":{"price":41.5475}},{"time":"2022-04-01T00:00:00Z","values":{"price":163.9375}},{"time":"2022-04-01T01:00:00Z","values":{"price":233.425}},{"time":"2022-04-01T02:00:00Z","values":{"price":67.3725}},{"time":"2022-04-01T03:00:00Z","values":{"price":43.1525}},{"time":"2022-04-01T04:00:00Z","values":{"price":34.715}},{"time":"2022-04-01T05:00:00Z","values":{"day_ahead_price":42.87,"mcpc_non_spin":2.72,"mcpc_regulation_down":7.01,"mcpc_regulation_up":6.85,"mcpc_responsive_reserve_service":6.85}},{"time":"2022-04-01T06:00:00Z","values":{"day_ahead_price":34.28,"mcpc_non_spin":2.72,"mcpc_regulation_down":5,"mcpc_regulation_up":5,"mcpc_responsive_reserve_service":5}},{"time":"2022-04-01T07:00:00Z","values":{"day_ahead_price":21.65,"mcpc_non_spin":3.45,"mcpc_regulation_down":4,"mcpc_regulation_up":4,"mcpc_responsive_reserve_service":4}},{"time":"2022-04-01T08:00:00Z","values":{"day_ahead_price":20.16,"mcpc_non_spin":3.45,"mcpc_regulation_down":5.71,"mcpc_regulation_up":5.71,"mcpc_responsive_reserve_service":5}},{"time":"2022-04-01T09:00:00Z","values":{"day_ahead_price":26.66,"mcpc_non_spin":3.1,"mcpc_regulation_down":6.85,"mcpc_regulation_up":6.85,"mcpc_responsive_reserve_service":6.27}},{"time":"2022-04-01T10:00:00Z","values":{"day_ahead_price":36.98,"mcpc_non_spin":3.8,"mcpc_regulation_down":12.85,"mcpc_regulation_up":12.85,"mcpc_responsive_reserve_service":11.42}},{"time":"2022-04-01T11:00:00Z","values":{"day_ahead_price":77.36,"mcpc_non_spin":29.77,"mcpc_regulation_down":13.5,"mcpc_regulation_up":34.15,"mcpc_responsive_reserve_service":34.15}},{"time":"2022-04-01T12:00:00Z","values":{"day_ahead_price":73.97,"mcpc_non_spin":21.29,"mcpc_regulation_down":16,"mcpc_regulation_up":20.29,"mcpc_responsive_reserve_service":20.29}},{"time":"2022-04-01T13:00:00Z","values":{"day_ahead_price":41.13,"mcpc_non_spin":11.7,"mcpc_regulation_down":13.5,"mcpc_regulation_up":13.5,"mcpc_responsive_reserve_service":11.7}},{"time":"2022-04-01T14:00:00Z","values":{"day_ahead_price":34.41,"mcpc_non_spin":10.85,"mcpc_regulation_down":14.75,"mcpc_regulation_up":14.75,"mcpc_responsive_reserve_service":10.85}},{"time":"2022-04-01T15:00:00Z","values":{"day_ahead_price":28.91,"mcpc_non_spin":4.42,"mcpc_regulation_down":13.7,"mcpc_regulation_up":12.85,"mcpc_responsive_reserve_service":10}},{"time":"2022-04-01T16:00:00Z","values":{"day_ahead_price":32.54,"mcpc_non_spin":4.42,"mcpc_regulation_down":13.64,"mcpc_regulation_up":13.64,"mcpc_responsive_reserve_service":9.74}},{"time":"2022-04-01T17:00:00Z","values":{"day_ahead_price":33.89,"mcpc_non_spin":4.62,"mcpc_regulation_down":14.05,"mcpc_regulation_up":12.26,"mcpc_responsive_reserve_service":10.15}},{"time":"2022-04-01T18:00:00Z","values":{"day_ahead_price":40.65,"mcpc_non_spin":10.2,"mcpc_regulation_down":13.57,"mcpc_regulation_up":11.55,"mcpc_responsive_reserve_service":10.2}},{"time":"2022-04-01T19:00:00Z","values":{"day_ahead_price":46.27,"mcpc_non_spin":7.15,"mcpc_regulation_down":13.87,"mcpc_regulation_up":11.85,"mcpc_responsive_reserve_service":10.37}},{"time":"2022-04-01T20:00:00Z","values":{"day_ahead_price":43.65,"mcpc_non_spin":6.39,"mcpc_regulation_down":15.99,"mcpc_regulation_up":13.97,"mcpc_responsive_reserve_service":12.98}},{"time":"2022-04-01T21:00:00Z","values":{"day_ahead_price":43.05,"mcpc_non_spin":5,"mcpc_regulation_down":18.87,"mcpc_regulation_up":16.85,"mcpc_responsive_reserve_service":15.37}},{"time":"2022-04-01T22:00:00Z","values":{"day_ahead_price":42.94,"mcpc_non_spin":5,"mcpc_regulation_down":20.54,"mcpc_regulation_up":18.52,"mcpc_responsive_reserve_service":17.92}},{"time":"2022-04-01T23:00:00Z","values":{"day_ahead_price":43.84,"mcpc_non_spin":5,"mcpc_regulation_down":14.66,"mcpc_regulation_up":12.64,"mcpc_responsive_reserve_service":11.21}},{"time":"2022-04-02T00:00:00Z","values":{"day_ahead_price":65.94,"mcpc_non_spin":18.65,"mcpc_regulation_down":20,"mcpc_regulation_up":24.39,"mcpc_responsive_reserve_service":24.39}},{"time":"2022-04-02T01:00:00Z","values":{"day_ahead_price":109.84,"mcpc_non_spin":53.95,"mcpc_regulation_down":20,"mcpc_regulation_up":45.73,"mcpc_responsive_reserve_service":49.73}},{"time":"2022-04-02T02:00:00Z","values":{"day_ahead_price":54.53,"mcpc_non_spin":8.3,"mcpc_regulation_down":8.54,"mcpc_regulation_up":9.04,"mcpc_responsive_reserve_service":9.04}},{"time":"2022-04-02T03:00:00Z","values":{"day_ahead_price":39.22,"mcpc_non_spin":4.02,"mcpc_regulation_down":6.15,"mcpc_regulation_up":5.65,"mcpc_responsive_reserve_service":5.65}},{"time":"2022-04-02T04:00:00Z","values":{"day_ahead_price":37.17,"mcpc_non_spin":3.25,"mcpc_regulation_down":5.84,"mcpc_regulation_up":5.77,"mcpc_responsive_reserve_service":5.77}}]}
    '''

    config.load_config('../../config.yml')
    # config.load_config('./config.yml')
    data = SimpleNamespace(**json.loads(esbr_input))
    # data.x = SimpleNamespace(**data.x)

    runner = FPSInference()
    output = runner.run(data)
    print(output)

