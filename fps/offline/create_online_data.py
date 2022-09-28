import pandas as pd
from datetime import datetime, timedelta
import json
from config import Config
import config
from common.connect_db import MariaDBConnection, MongoDBConnection
import yaml

from datetime import datetime, timedelta
from pytz import timezone, utc
from functools import reduce

import dateutil
import dill
from dateutil import tz
from dateutil.parser import parse

#%% Make DB Connection
with open('./config.yml', 'r') as c:
    config = yaml.load(c, Loader=yaml.FullLoader)


dev_conn_weather = MariaDBConnection(host=config['esbr_dev_conn']['host'],
                             port=config['esbr_dev_conn']['port'],
                             username=config['esbr_dev_conn']['username'],
                             password=config['esbr_dev_conn']['password'],
                             database='wunderground').get_db_conn()

dev_conn_ercot = MariaDBConnection(host=config['esbr_dev_conn']['host'],
                             port=config['esbr_dev_conn']['port'],
                             username=config['esbr_dev_conn']['username'],
                             password=config['esbr_dev_conn']['password'],
                             database='ercot').get_db_conn()

dev_conn_eo = MariaDBConnection(host=config['esbr_dev_conn']['host'],
                             port=config['esbr_dev_conn']['port'],
                             username=config['esbr_dev_conn']['username'],
                             password=config['esbr_dev_conn']['password'],
                             database='energyonline').get_db_conn()


esbr_summ_conn, esbr_summ_db = MongoDBConnection(host=config['esbr_conn']['host'],
                             port=config['esbr_conn']['port'],
                             username=config['esbr_conn']['username'],
                             password=config['esbr_conn']['password'],
                             database='summary').get_db_conn()

esbr_raw_conn, esbr_raw_db = MongoDBConnection(host=config['esbr_conn']['host'],
                             port=config['esbr_conn']['port'],
                             username=config['esbr_conn']['username'],
                             password=config['esbr_conn']['password'],
                             database='eip').get_db_conn()



#%%
with open('./config.yml', 'r') as c:
  config = yaml.load(c, Loader=yaml.FullLoader)

esbr_raw_conn, esbr_raw_db = MongoDBConnection(host=config['esbr_conn']['host'],
                             port=config['esbr_conn']['port'],
                             username=config['esbr_conn']['username'],
                             password=config['esbr_conn']['password'],
                             database='eip').get_db_conn()


master_id = '2022031423_CDT'
target = 'MVP_ERCOT_HOUSTON'

raw_collection = esbr_raw_db['MVP_HOUSTON']
docs = raw_collection.find({'master_id':master_id},
                       {'_id':0}).sort('date', 1)



#%% Inference #%%=================================================================================================
# 6013


inf_input = {
    "company": "EIP",
    "target": f"{target}",
    "service_type": "FPS",
    "input_case": "INFERENCE",
    "result_type": "REGRESSION",
    "master_id": master_id,
    "time": datetime.strftime(docs[0]['time'],'%Y-%m-%dT%H:%M:%SZ')}

inf_input_x = []
for x_temp in docs[0]['x']:
    inf_input_x.append({'time': datetime.strftime(x_temp['time'], '%Y-%m-%dT%H:%M:%SZ'),
                        'values': x_temp['values']})


#%% Add X
local_tz = timezone('US/Central')
start_local = '2022-01-13 00:00:00'
end_local = '2022-01-13 23:00:00'
start_utc = local_tz.localize(dateutil.parser.parse(start_local)).astimezone(utc)
end_utc = local_tz.localize(dateutil.parser.parse(end_local)).astimezone(utc)



### day-ahead
day_ahead_query = f"""SELECT date, price AS day_ahead_price
            FROM ercot_day_ahead_price
            WHERE date >= '{start_utc}' AND date <= '{end_utc}' AND zone='LZ_HOUSTON'"""
day_ahead_raw = pd.read_sql(day_ahead_query, dev_conn_eo)



### ercot_ancillary_services_mcp
anci_query = f"""SELECT date, mcpc, ancillarytype
              FROM ercot_ancillary_services_mcp
              WHERE date >= '{start_utc}' AND date <= '{end_utc}'
              """
anci_raw = pd.read_sql(anci_query, dev_conn_eo)

#%%
class XPreprocessing:
  def __init__(self, start_utc: datetime, end_utc: datetime, local_tz: timezone):
    self.start_utc = start_utc
    self.end_utc = end_utc
    self.local_tz = local_tz
    self.utc_time_col = 'date'
    self.local_time_col = 'local_date'

  def _make_local_time(self, data):
    result = data.copy()
    result[self.local_time_col] = result[self.utc_time_col].apply(lambda x: x.astimezone(self.local_tz))
    result.drop(self.utc_time_col, axis=1, inplace=True)
    return result

  def preprocess_actual_load(self, data: pd.DataFrame) -> pd.DataFrame:
    data.columns = data.columns.str.replace('HOUSTON', 'load')
    data[self.utc_time_col] = pd.to_datetime(data[self.utc_time_col], utc=True)
    result = data.loc[(data[self.utc_time_col] >= self.start_utc) & (data[self.utc_time_col] <= self.end_utc), :]
    result = self._make_local_time(result)
    return result

  def preprocess_weather_historical(self, data: pd.DataFrame) -> pd.DataFrame:
    data.columns = data.columns.str.replace('time', self.utc_time_col)
    data[self.utc_time_col] = pd.to_datetime(data[self.utc_time_col], utc=True)

    data.insert(0, 'date_temp', data[self.utc_time_col].apply(lambda x: x.round(freq='H')))
    data = data.groupby(['date_temp']).first().reset_index()
    result = data.drop([self.utc_time_col], axis=1).rename(columns={'date_temp': self.utc_time_col})
    result = self._make_local_time(result)
    return result

  def preprocess_day_ahead(self, data: pd.DataFrame) -> pd.DataFrame:
    data[self.utc_time_col] = pd.to_datetime(data[self.utc_time_col], utc=True)
    result = self._make_local_time(data)
    return result

  def preprocess_ancillary_mcpc(self, data: pd.DataFrame) -> pd.DataFrame:
    prefix = 'mcpc'
    anci_type_dict = {'Non Spin': f'{prefix}_non_spin',
                      'Regulation Down': f'{prefix}_reg_down',
                      'Regulation Up': f'{prefix}_reg_up',
                      'Responsive Reserve Service': f'{prefix}_rrs'}
    data['ancillarytype'] = data['ancillarytype'].apply(lambda x: anci_type_dict[x])
    result = pd.pivot_table(data, values='mcpc', index='date', columns='ancillarytype')
    result = result.reset_index()
    result[self.utc_time_col] = pd.to_datetime(result[self.utc_time_col], utc=True)
    result = self._make_local_time(result)
    return result

  def merge_x(self, x_data_list: list) -> pd.DataFrame:
    result = reduce(lambda left, right: pd.merge(left, right, on=self.local_time_col,
                                                 how='outer'), x_data_list).dropna()
    return result



day_ahead_df = day_ahead_raw.copy()
anci_df = anci_raw.copy()

x_prep = XPreprocessing(start_utc, end_utc, local_tz)

day_ahead_df = x_prep.preprocess_day_ahead(day_ahead_df)
anci_df = x_prep.preprocess_ancillary_mcpc(anci_df)

# merge
x_list = [day_ahead_df, anci_df]
x_list = [x for x in x_list if not x.empty]
x_merged = x_prep.merge_x(x_list)

x_merged['date'] = x_merged['local_date'].apply(lambda x: x.astimezone(utc))


#%%
x_val_dict = x_merged.loc[:, ['day_ahead_price','mcpc_non_spin', 'mcpc_reg_down',
       'mcpc_reg_up', 'mcpc_rrs']].to_dict(orient='index')
x_date_dict = x_merged.loc[:, ['date']].to_dict(orient='index')


x_temp_list = []
for i, val in enumerate(x_val_dict.values()):

  x_temp = {'time': datetime.strftime(x_date_dict[i]['date'], '%Y-%m-%dT%H:%M:%SZ'),
            'values': val}
  x_temp_list.append(x_temp)



#%%
inf_input_x = inf_input_x + x_temp_list
inf_input['x'] = inf_input_x
inf_input_str = json.dumps(inf_input)
#%%


#%%
x_temp_list = [
  {
    "date": "2021-12-25 00:00:00",
    "price": 30.405
  },
  {
    "date": "2021-12-25 01:00:00",
    "price": 29.37
  },
  {
    "date": "2021-12-25 02:00:00",
    "price": 29.4
  },
  {
    "date": "2021-12-25 03:00:00",
    "price": 29.665
  },
  {
    "date": "2021-12-25 04:00:00",
    "price": 38.3525
  },
  {
    "date": "2021-12-25 05:00:00",
    "price": 46.1575
  },
  {
    "date": "2021-12-25 06:00:00",
    "price": 30.6375
  },
  {
    "date": "2021-12-25 07:00:00",
    "price": 36.8275
  },
  {
    "date": "2021-12-25 08:00:00",
    "price": 44.4525
  },
  {
    "date": "2021-12-25 09:00:00",
    "price": 30.2975
  },
  {
    "date": "2021-12-25 10:00:00",
    "price": 56.295
  },
  {
    "date": "2021-12-25 11:00:00",
    "price": 109.0425
  },
  {
    "date": "2021-12-25 12:00:00",
    "price": 30.295
  },
  {
    "date": "2021-12-25 13:00:00",
    "price": 30.635
  },
  {
    "date": "2021-12-25 14:00:00",
    "price": 31.76
  },
  {
    "date": "2021-12-25 15:00:00",
    "price": 27.58
  },
  {
    "date": "2021-12-25 16:00:00",
    "price": 26.8225
  },
  {
    "date": "2021-12-25 17:00:00",
    "price": 31.68
  },
  {
    "date": "2021-12-25 18:00:00",
    "price": 30.4175
  },
  {
    "date": "2021-12-25 19:00:00",
    "price": 28.9625
  },
  {
    "date": "2021-12-25 20:00:00",
    "price": 28.55
  },
  {
    "date": "2021-12-25 21:00:00",
    "price": 26.22
  },
  {
    "date": "2021-12-25 22:00:00",
    "price": 31.405
  },
  {
    "date": "2021-12-25 23:00:00",
    "price": 30.0325
  },
  {
    "date": "2021-12-26 00:00:00",
    "price": 26.6425
  },
  {
    "date": "2021-12-26 01:00:00",
    "price": 17.9125
  },
  {
    "date": "2021-12-26 02:00:00",
    "price": 2.74
  },
  {
    "date": "2021-12-26 03:00:00",
    "price": -4.325
  },
  {
    "date": "2021-12-26 04:00:00",
    "price": -4.665
  },
  {
    "date": "2021-12-26 05:00:00",
    "price": -5.3525
  },
  {
    "date": "2021-12-26 06:00:00",
    "price": -4.5075
  },
  {
    "date": "2021-12-26 07:00:00",
    "price": -3.3825
  },
  {
    "date": "2021-12-26 08:00:00",
    "price": 2.9825
  },
  {
    "date": "2021-12-26 09:00:00",
    "price": 18.9425
  },
  {
    "date": "2021-12-26 10:00:00",
    "price": 13.9575
  },
  {
    "date": "2021-12-26 11:00:00",
    "price": 11.87
  },
  {
    "date": "2021-12-26 12:00:00",
    "price": 17.3225
  },
  {
    "date": "2021-12-26 13:00:00",
    "price": 16.5075
  },
  {
    "date": "2021-12-26 14:00:00",
    "price": 18.225
  },
  {
    "date": "2021-12-26 15:00:00",
    "price": 14.6175
  },
  {
    "date": "2021-12-26 16:00:00",
    "price": 19.0375
  },
  {
    "date": "2021-12-26 17:00:00",
    "price": 22.625
  },
  {
    "date": "2021-12-26 18:00:00",
    "price": 22.1875
  },
  {
    "date": "2021-12-26 19:00:00",
    "price": 19.67
  },
  {
    "date": "2021-12-26 20:00:00",
    "price": 18.9725
  },
  {
    "date": "2021-12-26 21:00:00",
    "price": 21.7925
  },
  {
    "date": "2021-12-26 22:00:00",
    "price": 24.1975
  },
  {
    "date": "2021-12-26 23:00:00",
    "price": 18.52
  },
  {
    "date": "2021-12-27 00:00:00",
    "price": 17.0575
  },
  {
    "date": "2021-12-27 01:00:00",
    "price": 17.0975
  },
  {
    "date": "2021-12-27 02:00:00",
    "price": 17.16
  },
  {
    "date": "2021-12-27 03:00:00",
    "price": 22.685
  },
  {
    "date": "2021-12-27 04:00:00",
    "price": 24.8975
  },
  {
    "date": "2021-12-27 05:00:00",
    "price": 25.79
  },
  {
    "date": "2021-12-27 06:00:00",
    "price": 23.4975
  },
  {
    "date": "2021-12-27 07:00:00",
    "price": 28.715
  },
  {
    "date": "2021-12-27 08:00:00",
    "price": 28.38
  },
  {
    "date": "2021-12-27 09:00:00",
    "price": 28.1825
  },
  {
    "date": "2021-12-27 10:00:00",
    "price": 31.57
  },
  {
    "date": "2021-12-27 11:00:00",
    "price": 31.355
  },
  {
    "date": "2021-12-27 12:00:00",
    "price": 32.5675
  },
  {
    "date": "2021-12-27 13:00:00",
    "price": 34.1275
  },
  {
    "date": "2021-12-27 14:00:00",
    "price": 33.35
  },
  {
    "date": "2021-12-27 15:00:00",
    "price": 35.0525
  },
  {
    "date": "2021-12-27 16:00:00",
    "price": 37.12
  },
  {
    "date": "2021-12-27 17:00:00",
    "price": 45.4675
  },
  {
    "date": "2021-12-27 18:00:00",
    "price": 32.895
  },
  {
    "date": "2021-12-27 19:00:00",
    "price": 30.55
  },
  {
    "date": "2021-12-27 20:00:00",
    "price": 24.7625
  },
  {
    "date": "2021-12-27 21:00:00",
    "price": 23.5175
  },
  {
    "date": "2021-12-27 22:00:00",
    "price": 23.97
  },
  {
    "date": "2021-12-27 23:00:00",
    "price": 22.36
  },
  {
    "date": "2021-12-28 00:00:00",
    "price": 19.615
  },
  {
    "date": "2021-12-28 01:00:00",
    "price": 12.3575
  },
  {
    "date": "2021-12-28 02:00:00",
    "price": 0.48
  },
  {
    "date": "2021-12-28 03:00:00",
    "price": 3.4125
  },
  {
    "date": "2021-12-28 04:00:00",
    "price": 18.465
  },
  {
    "date": "2021-12-28 05:00:00",
    "price": 29.33
  },
  {
    "date": "2021-12-28 06:00:00",
    "price": 24.2125
  },
  {
    "date": "2021-12-28 07:00:00",
    "price": 22.9475
  },
  {
    "date": "2021-12-28 08:00:00",
    "price": 22.68
  },
  {
    "date": "2021-12-28 09:00:00",
    "price": 31.9
  },
  {
    "date": "2021-12-28 10:00:00",
    "price": 21.19
  },
  {
    "date": "2021-12-28 11:00:00",
    "price": 22.5375
  },
  {
    "date": "2021-12-28 12:00:00",
    "price": 22.255
  },
  {
    "date": "2021-12-28 13:00:00",
    "price": 23.7875
  },
  {
    "date": "2021-12-28 14:00:00",
    "price": 23.465
  },
  {
    "date": "2021-12-28 15:00:00",
    "price": 23.4275
  },
  {
    "date": "2021-12-28 16:00:00",
    "price": 27.675
  },
  {
    "date": "2021-12-28 17:00:00",
    "price": 521.49
  },
  {
    "date": "2021-12-28 18:00:00",
    "price": 60.675
  },
  {
    "date": "2021-12-28 19:00:00",
    "price": 49.5275
  },
  {
    "date": "2021-12-28 20:00:00",
    "price": 38.015
  },
  {
    "date": "2021-12-28 21:00:00",
    "price": 30.7625
  },
  {
    "date": "2021-12-28 22:00:00",
    "price": 30.4
  },
  {
    "date": "2021-12-28 23:00:00",
    "price": 29.2125
  },
  {
    "date": "2021-12-29 00:00:00",
    "price": 27.0375
  },
  {
    "date": "2021-12-29 01:00:00",
    "price": 23.42
  },
  {
    "date": "2021-12-29 02:00:00",
    "price": 22.53
  },
  {
    "date": "2021-12-29 03:00:00",
    "price": 23.125
  },
  {
    "date": "2021-12-29 04:00:00",
    "price": 31.87
  },
  {
    "date": "2021-12-29 05:00:00",
    "price": 35.6975
  },
  {
    "date": "2021-12-29 06:00:00",
    "price": 36.22
  },
  {
    "date": "2021-12-29 07:00:00",
    "price": 36.0675
  },
  {
    "date": "2021-12-29 08:00:00",
    "price": 30.8025
  },
  {
    "date": "2021-12-29 09:00:00",
    "price": 23.9025
  },
  {
    "date": "2021-12-29 10:00:00",
    "price": 23.39
  },
  {
    "date": "2021-12-29 11:00:00",
    "price": 25.5325
  },
  {
    "date": "2021-12-29 12:00:00",
    "price": 24.9525
  },
  {
    "date": "2021-12-29 13:00:00",
    "price": 29.705
  },
  {
    "date": "2021-12-29 14:00:00",
    "price": 32.8975
  },
  {
    "date": "2021-12-29 15:00:00",
    "price": 25.925
  },
  {
    "date": "2021-12-29 16:00:00",
    "price": 24.2425
  },
  {
    "date": "2021-12-29 17:00:00",
    "price": 133.7075
  },
  {
    "date": "2021-12-29 18:00:00",
    "price": 76.3775
  },
  {
    "date": "2021-12-29 19:00:00",
    "price": 48.015
  },
  {
    "date": "2021-12-29 20:00:00",
    "price": 43.8575
  },
  {
    "date": "2021-12-29 21:00:00",
    "price": 34.155
  },
  {
    "date": "2021-12-29 22:00:00",
    "price": 52.1225
  },
  {
    "date": "2021-12-29 23:00:00",
    "price": 27.9675
  },
  {
    "date": "2021-12-30 00:00:00",
    "price": 51.0075
  },
  {
    "date": "2021-12-30 01:00:00",
    "price": 38.315
  },
  {
    "date": "2021-12-30 02:00:00",
    "price": 26.5075
  },
  {
    "date": "2021-12-30 03:00:00",
    "price": 25.945
  },
  {
    "date": "2021-12-30 04:00:00",
    "price": 30.3625
  },
  {
    "date": "2021-12-30 05:00:00",
    "price": 27.0325
  },
  {
    "date": "2021-12-30 06:00:00",
    "price": 22.79
  },
  {
    "date": "2021-12-30 07:00:00",
    "price": 24.07
  },
  {
    "date": "2021-12-30 08:00:00",
    "price": 23.6
  },
  {
    "date": "2021-12-30 09:00:00",
    "price": 23.6075
  },
  {
    "date": "2021-12-30 10:00:00",
    "price": 32.235
  },
  {
    "date": "2021-12-30 11:00:00",
    "price": 30.1075
  },
  {
    "date": "2021-12-30 12:00:00",
    "price": 33.7675
  },
  {
    "date": "2021-12-30 13:00:00",
    "price": 34.01
  },
  {
    "date": "2021-12-30 14:00:00",
    "price": 26.015
  },
  {
    "date": "2021-12-30 15:00:00",
    "price": 25.3075
  },
  {
    "date": "2021-12-30 16:00:00",
    "price": 31.5025
  },
  {
    "date": "2021-12-30 17:00:00",
    "price": 63.34
  },
  {
    "date": "2021-12-30 18:00:00",
    "price": 32.425
  },
  {
    "date": "2021-12-30 19:00:00",
    "price": 33.7975
  },
  {
    "date": "2021-12-30 20:00:00",
    "price": 26.34
  },
  {
    "date": "2021-12-30 21:00:00",
    "price": 23.5775
  },
  {
    "date": "2021-12-30 22:00:00",
    "price": 22.5925
  },
  {
    "date": "2021-12-30 23:00:00",
    "price": 22.9025
  },
  {
    "date": "2021-12-31 00:00:00",
    "price": 19.5425
  },
  {
    "date": "2021-12-31 01:00:00",
    "price": 17.55
  },
  {
    "date": "2021-12-31 02:00:00",
    "price": 17.545
  },
  {
    "date": "2021-12-31 03:00:00",
    "price": 19.7675
  },
  {
    "date": "2021-12-31 04:00:00",
    "price": 22.56
  },
  {
    "date": "2021-12-31 05:00:00",
    "price": 22.64
  },
  {
    "date": "2021-12-31 06:00:00",
    "price": 23.21
  },
  {
    "date": "2021-12-31 07:00:00",
    "price": 26.65
  },
  {
    "date": "2021-12-31 08:00:00",
    "price": 24.9675
  },
  {
    "date": "2021-12-31 09:00:00",
    "price": 27.9275
  },
  {
    "date": "2021-12-31 10:00:00",
    "price": 28.9025
  },
  {
    "date": "2021-12-31 11:00:00",
    "price": 30.39
  },
  {
    "date": "2021-12-31 12:00:00",
    "price": 35.275
  },
  {
    "date": "2021-12-31 13:00:00",
    "price": 41.5125
  },
  {
    "date": "2021-12-31 14:00:00",
    "price": 40.995
  },
  {
    "date": "2021-12-31 15:00:00",
    "price": 35.7375
  },
  {
    "date": "2021-12-31 16:00:00",
    "price": 27.8775
  },
  {
    "date": "2021-12-31 17:00:00",
    "price": 23.81
  },
  {
    "date": "2021-12-31 18:00:00",
    "price": 23.2075
  },
  {
    "date": "2021-12-31 19:00:00",
    "price": 24.8325
  },
  {
    "date": "2021-12-31 20:00:00",
    "price": 34.21
  },
  {
    "date": "2021-12-31 21:00:00",
    "price": 49.0325
  },
  {
    "date": "2021-12-31 22:00:00",
    "price": 59.3675
  },
  {
    "date": "2021-12-31 23:00:00",
    "price": 64.115
  }
]

#%%
inf_input_x = []
for x_temp in x_temp_list:
    inf_input_x.append({'time': datetime.strftime(datetime.strptime(x_temp['date'],'%Y-%m-%d %H:%M:%S')+timedelta(hours=6),
                                                  '%Y-%m-%dT%H:%M:%SZ'),
    'values': {'price' : x_temp['price']}})

#%%
inf_input['x'] = inf_input_x

#%%
inf_input_str = json.dumps(inf_input)



#%% Feedback #%%=================================================================================================

feedback_input = {
    "company": "EIP",
    "target": "ERCOT_HOUSTON",
    "service_type": "FPS",
    "input_case": "FEEDBACK",
    "result_type": "REGRESSION",
    "master_id": "2022010110_CST",
    "y": {
        "time": "2021-12-31T18:00:00.000Z",
        "value": 21.33
    },
    "benchmark": {
        "time": "2021-12-31T18:00:00.000Z",
        "value": 24.45
    }
}

feedback_input_str = json.dumps(feedback_input)



#%% Feedback #%%=================================================================================================
update_input = {
    "company": "EIP",
    "target": "ERCOT_HOUSTON",
    "service_type": "FPS",
    "input_case": "MODEL_UPDATE",
    "result_type": "REGRESSION",
    "master_id": "2021091514_CDT",
}

update_input_str = json.dumps(update_input)
