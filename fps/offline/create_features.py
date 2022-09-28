
import pandas as pd
from datetime import datetime, timedelta
from pytz import timezone, utc

import dateutil
from dateutil.parser import parse

import importlib
import models
importlib.reload(models)

from models.preprocess.preprocessing_y import CreateDate, SummaryData
from models.preprocess import load_x
from models.preprocess.preprocessing_x import XPreprocessing
from models.conn_obj import DBConn

#%% Docs
"""
Summary 데이터 생성
start_local, end_local 에 분석 지역 기준 로컬 시간을 입력
=> X크롤링 데이터 및 y데이터 로드하여 Summary 생성 후 저장
"""

#%% Setting
company = 'EIP'
# target = 'ERCOT_HOUSTON'
target = 'MVP_ERCOT_HOUSTON'
summ_cnt = 100
local_tz = timezone('US/Central')

start_local = '2021-12-01 00:00:00'
end_local = '2022-05-23 00:00:00'
# end_local = dateutil.parser.parse(start_local) + timedelta(days=summ_cnt)#'2022-01-14 23:00:00'
start_utc = local_tz.localize(dateutil.parser.parse(start_local)).astimezone(utc)
end_utc = local_tz.localize(dateutil.parser.parse(end_local)).astimezone(utc)

y_start = dateutil.parser.parse(start_local) - timedelta(days=7)# D-7 변수 생성을 위해
y_start_utc = local_tz.localize(y_start).astimezone(utc)

conn = DBConn()

#%% Load X data

load_df = load_x.load_data_from_db('load', start_utc, end_utc)
weather_df = load_x.load_data_from_db('weather', start_utc, end_utc)
day_ahead_df = load_x.load_data_from_db('day_ahead', start_utc, end_utc)
anci_df = load_x.load_data_from_db('ancillary_services_mcp', start_utc, end_utc)

#%% Preprocessing X
x_prep = XPreprocessing(start_utc, end_utc, local_tz)

load_df = x_prep.preprocess_actual_load(load_df)
weather_df = x_prep.preprocess_weather_historical(weather_df)
day_ahead_df = x_prep.preprocess_day_ahead(day_ahead_df)
anci_df = x_prep.preprocess_ancillary_mcpc(anci_df)


#%% merge
def create_pow_feats(data):
    pow_data = data.apply(lambda x: x**2)
    pow_data.columns = [x + '_pow' for x in data.columns]
    return pow_data


x_list = [day_ahead_df, load_df, weather_df, anci_df]
x_merged = x_prep.merge_x(x_list)

x_merge_temp = x_merged.drop('local_date', axis=1)
x_merge_pow = create_pow_feats(x_merge_temp)


x_merged = pd.concat([x_merged, x_merge_pow], axis=1)

#%% Load y data from crawlled data
#

def get_master_id(x, local_tz):
    x = x.replace(tzinfo=utc).astimezone(local_tz)
    zone_name = x.tzname()
    time_info = datetime.strftime(x, '%Y%m%d%H')
    id = time_info + '_' + zone_name
    return id



y_start_utc = start_utc - timedelta(days=7)
y_query = f'''SELECT date, price AS y FROM ercot_real_time_price
            WHERE date >="{y_start_utc}" AND date <="{end_utc}"
            AND zone = "LZ_HOUSTON"'''

y_raw = pd.read_sql(y_query, conn.dev_conn_eo)
y_df = y_raw.copy()

y_df['master_id'] = y_df['date'].apply(lambda x: get_master_id(x, local_tz))

date_obj = CreateDate('date', local_tz)
y_temp = date_obj.transform(y_df)

summ_obj = SummaryData('y', 'local_date')
y_summ = summ_obj.transform(y_temp, use_summ=True)

y_pow_cols = ['D_1', 'D_2', 'D_3', 'D_4', 'D_5', 'D_6', 'D_7',
              'day_before_min', 'day_before_max', 'day_before_avg']

y_pow_df = create_pow_feats(y_summ.loc[:, y_pow_cols])
y_summ = pd.concat([y_summ, y_pow_df], axis=1)



#%%  Merge
total_merged = pd.merge(y_summ, x_merged, on='local_date', how='outer')
total_merged['local_date'] = total_merged['local_date'].apply(lambda x: datetime.strftime(x, '%Y-%m-%d %H:%M:%S'))

#%% Save

def save_data_to_db(collection, data):
    for label, content in data.iterrows():
        data_dict = dict(content)
        resp = collection.update_one({'master_id': content['master_id']},
                               {'$set': data_dict}, upsert=True)
        print(f'{label}:{resp.raw_result}')

summ_collection = conn.esbr_summ_db[target]
save_data_to_db(summ_collection, total_merged)