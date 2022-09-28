from types import SimpleNamespace
import pandas as pd
import json
from datetime import datetime, timedelta
from dateutil.parser import parse as time_parse
import traceback
from pytz import utc
from pymongo.collection import Collection
import random
random.seed(42)

from common.error import make_error_msg
from common.handler import Handler
from qpslib_history_manager import history_client
from common import evaluation
from models.conn_obj import DBConn
from models.preprocess import features, load_x
from models.preprocess.preprocessing_x import XPreprocessing
from models.conf import EvaluationItems, ModelParams

from config import Config
import config


import qpslib_history_manager
qpslib_history_manager.__version__
#%%


#%%

class FPSFeedback(Handler):
    def __init__(self):
        cfg = Config()
        self._service_host = cfg.get_service_host()
        self.local_tz = cfg.get_timezone()
        self.stacked_model = None
        self.summ_date_name = ModelParams.local_date
        self.dayname_dict = EvaluationItems.dayname_dict
        self.month_dict = EvaluationItems.month_dict
        self.raw_db_time = 'time'
        self.outlier = ModelParams.outlier
        

    # @concurrent.process(timeout=30) # not work with
    def run(self, data: SimpleNamespace):
        # setting
        company = data.company
        target = data.target
        master_id = data.master_id
        result_type = data.result_type
        y = data.y.value
        benchmark = data.benchmark.value
        predict_time = time_parse(data.y.time).astimezone(self.local_tz)
        predict_time_str = datetime.strftime(predict_time, '%Y-%m-%d %H:%M:%S')
        client = history_client.FPSHistoryClient(self._service_host, company, target, result_type)

        # db seting
        conn = DBConn()
        raw_collection = conn.esbr_raw_db[target]
        inf_collection = conn.esbr_history_db['inferences']
        summ_collection = conn.esbr_summ_db[target]
        service_start, service_start_str = self._get_service_start(inf_collection, target)

        # output initiallize
        output = {}
        comment = ''
        print(f'Start Feedback : {master_id}')

        # ## test
        # y_hat = 43.444927

        # load yhat
        try:
            y_hat = client.get_inference_result(master_id).y_hat
            if y_hat == None:
                raise Exception("Yhat is None")
            output['y'] = y
            output['benchmark'] = benchmark
        except Exception as e:
            output['error'] = make_error_msg(str(e), f'Load yhat: {traceback.format_exc()}')
            print(f'output: {output}')
            output = self._call_last_result_for_error_case(output, client)
            return output

        # get residual
        try:
            residual = evaluation.cal_residual(y, y_hat)
            output['residual'] = residual
        except Exception as e:
            output['error'] = make_error_msg(str(e), f'Calculate residual: {traceback.format_exc()}')
            print(f'output: {output}')
            output = self._call_last_result_for_error_case(output, client)
            return output

        # load history x crawled data
        try:
            x_local_start = predict_time - timedelta(minutes=30)
            x_local_end = predict_time + timedelta(minutes=30)
            x_utc_start = x_local_start.astimezone(utc)
            x_utc_end = x_local_end.astimezone(utc)
            load_df = load_x.load_data_from_db('load', x_utc_start, x_utc_end)
            weather_df = load_x.load_data_from_db('weather', x_utc_start, x_utc_end)
            x_prep = XPreprocessing(x_utc_start, x_utc_end, self.local_tz)
            load_df = x_prep.preprocess_actual_load(load_df)
            weather_df = x_prep.preprocess_weather_historical(weather_df)
            if not weather_df.empty:
                weather_df = weather_df.loc[weather_df[self.summ_date_name].dt.hour == predict_time.hour, :]

            x_list = [df for df in [load_df, weather_df] if not df.empty]
            if x_list:
                x_merged = x_prep.merge_x(x_list)
            else:
                x_merged = pd.DataFrame(columns=[self.summ_date_name])
        except Exception as e:
            output['error'] = make_error_msg(str(e), f'Create historical x features: {traceback.format_exc()}')
            print(f'output: {output}')
            output = self._call_last_result_for_error_case(output, client)
            return output

        # merge x & y
        try:
            y_df = pd.DataFrame({'master_id': master_id,  'y': y, self.summ_date_name: predict_time}, index=[0])
            total_df = pd.merge(y_df, x_merged, on=self.summ_date_name)
            total_df[self.summ_date_name] = total_df[self.summ_date_name].apply(
                lambda x: datetime.strftime(x, '%Y-%m-%d %H:%M:%S'))
        except Exception as e:
            output['error'] = make_error_msg(str(e), f'Merge x features and y: {traceback.format_exc()}')
            print(f'output: {output}')
            output = self._call_last_result_for_error_case(output, client)
            return output

        # save featurs to summary
        try:
            features.save_features_to_db(summ_collection, total_df)
        except Exception as e:
            output['error'] = make_error_msg(str(e), f'Save y to summary DB: {traceback.format_exc()}')
            print(f'output: {output}')
            output = self._call_last_result_for_error_case(output, client)
            return output

        # Performance ====================================================================
        # Load all records
        try:
            y_records = self.load_raw_data('y', raw_collection, service_start, predict_time)
            benchmark_records = self.load_raw_data('benchmark', raw_collection, service_start, predict_time)
            raw_data = pd.merge(y_records, benchmark_records, on=['master_id', self.raw_db_time])
            raw_data = raw_data.dropna().reset_index(drop=True)
            check_id_list = raw_data['master_id'].values.tolist()
            yhat_records = self.load_inference_data(target, inf_collection, check_id_list)
            check_data = pd.merge(raw_data, yhat_records, on='master_id')
            check_data = check_data.dropna().reset_index(drop=True)
            check_data[self.raw_db_time] = pd.to_datetime(check_data[self.raw_db_time], utc=True)
        except Exception as e:
            output['error'] = make_error_msg(str(e), f'Load y, yhat: {traceback.format_exc()}')
            print(f'output: {output}')
            output = self._call_last_result_for_error_case(output, client)
            return output

        # get performance
        try:
            hourly = self.get_hourly_performance(check_data, 'y', 'y_hat')
            b_hourly = self.get_hourly_performance(check_data, 'y', 'benchmark')
            weekly = self.get_weekly_performance(check_data, 'y', 'y_hat')
            b_weekly = self.get_weekly_performance(check_data, 'y', 'benchmark')
            monthly = self.get_montly_performance(check_data, 'y', 'y_hat')
            b_monthly = self.get_montly_performance(check_data, 'y', 'benchmark')
            total = self.get_total_performance(service_start_str, predict_time_str,
                                               check_data, 'y', 'y_hat')
            performance = {'hourly': hourly,
                          'weekly': weekly,
                          'monthly': monthly}
            b_performance = {'hourly': b_hourly,
                           'weekly': b_weekly,
                           'monthly': b_monthly}
        except Exception as e:
            output['error'] = make_error_msg(str(e), f'Get performance : {traceback.format_exc()}')
            print(f'output: {output}')
            output = self._call_last_result_for_error_case(output, client)
            return output

        # Define output ====================================================================
        output['total'] = total
        output['performance'] = performance
        output['benchmark_performance'] = b_performance
        output['comment'] = comment


        return output

    def _call_last_result_for_error_case(self, output: dict, client: history_client) -> dict:
        last_result = client.get_last_feedback_result()
        output['performance'] = last_result.performance
        output['benchmark_performance'] = last_result.benchmark_performance
        output['total'] = last_result.total
        return output

    def _get_service_start(self, collection: Collection, target: str) -> str:
        docs = collection.find({'target': target}, {'_id': 0, 'master_id': 1}).\
            sort('master_id', 1).limit(1)
        start_master_id = docs[0]['master_id']
        start_time = datetime.strptime(start_master_id.split('_')[0], '%Y%m%d%H%M')
        start_time_str = datetime.strftime(start_time, '%Y-%m-%d %H:%M:%S')
        return start_time, start_time_str


    def get_hourly_performance(self, check_data: pd.DataFrame, y_name: str, check_name: str) -> dict:
        data = check_data.copy()
        data['hour'] = data[self.raw_db_time].apply(lambda x: str(x.astimezone(self.local_tz).hour).zfill(2))
        hourly = {str(x).zfill(2): {'value': 0, 'count': 0} for x in range(24)}
        for hour in hourly.keys():
            check = data.loc[data['hour'] == hour, :]
            performance = evaluation.cal_100_smape(check[y_name], check[check_name], self.outlier)
            hourly[hour]['value'] = performance
            hourly[hour]['count'] = len(check)
        return hourly


    def get_weekly_performance(self, check_data: pd.DataFrame, y_name: str, check_name: str) -> dict:
        data = check_data.copy()
        data['week'] = data[self.raw_db_time].apply(lambda x: x.astimezone(self.local_tz).strftime('%A')[:3])
        weekly = {self.dayname_dict[x]: {'value': 0, 'count': 0} for x in range(7)}
        for week in weekly.keys():
            check = data.loc[data['week'] == week, :]
            if check.empty:
                weekly[week]['value'] = 0
                weekly[week]['count'] = 0
            else:
                performance = evaluation.cal_100_smape(check[y_name], check[check_name], self.outlier)
                weekly[week]['value'] = performance
                weekly[week]['count'] = len(check)
        return weekly

    def get_montly_performance(self, check_data: pd.DataFrame, y_name: str, check_name: str) -> dict:
        data = check_data.copy()
        data['month'] = data[self.raw_db_time].apply(lambda x: self.month_dict[x.astimezone(self.local_tz).month])
        monthly = {self.month_dict[x]: {'value': 0, 'count': 0} for x in range(1, 13)}
        for month in monthly.keys():
            check = data.loc[data['month'] == month, :]
            if check.empty:
                monthly[month]['value'] = 0
                monthly[month]['count'] = 0
            else:
                performance = evaluation.cal_100_smape(check[y_name], check[check_name], self.outlier)
                monthly[month]['value'] = performance
                monthly[month]['count'] = len(check)
        return monthly

    def get_total_performance(self, local_start: str, local_end: str, check_data: pd.DataFrame,
                              y_name: str, check_name: str) -> dict:
        check = check_data.copy()
        performance = evaluation.cal_100_smape(check[y_name], check[check_name], self.outlier)
        result = {'duration':
                      {'start': local_start,
                       'end': local_end},
                  'value': performance,
                  'count': len(check)}
        return result


    def load_raw_data(self, document: str, collection: Collection, start_local:datetime, end_local:datetime) -> pd.DataFrame:
        # utc_start = self.local_tz.localize(time_parse(start_local)).astimezone(utc)
        # utc_end = end_local.astimezone(utc)
        utc_start = start_local.astimezone(utc)
        utc_end = end_local.astimezone(utc)

        docs = collection.find({self.raw_db_time: {'$gte': utc_start, '$lte': utc_end}},
                               {'_id': 0, 'master_id': 1, self.raw_db_time: 1, document: 1}).sort(self.raw_db_time, 1)
        result = pd.DataFrame(docs)
        result[document] = result[document].apply(lambda x: x['value'])
        result = result.dropna().reset_index(drop=True)
        return result


    def load_inference_data(self, target: str, collection: Collection, id_list: list) -> pd.DataFrame:
        docs = collection.find({'target': target, 'master_id': {'$in':id_list}},
                               {'_id': 0, 'master_id': 1, 'y_hat': 1})
        result = pd.DataFrame(docs)
        result = result.dropna().reset_index(drop=True)
        return result


    def save_y_to_summ_db(self, collection, master_id, y):
        resp = collection.update_one({'master_id': master_id},
                               {'$set': {'y': y}}, upsert=True)
        print(f'{master_id}:{resp.raw_result}')





if __name__ == '__main__':
    esbr_input = '''
  {"company":"EIP","target":"MVP_ERCOT_HOUSTON","service_type":"FPS","input_case":"FEEDBACK","result_type":"REGRESSION","master_id":"2022052522_CDT","y":{"time":"2022-05-26T03:00:00Z","value":106.675},"benchmark":{"time":"2022-05-26T03:00:00Z","value":45.655}}
    '''

    config.load_config('../../config.yml')

    data = SimpleNamespace(**json.loads(esbr_input))
    data.y = SimpleNamespace(**data.y)
    data.benchmark = SimpleNamespace(**data.benchmark)

    runner = FPSFeedback()
    output = runner.run(data)
    print(output)

