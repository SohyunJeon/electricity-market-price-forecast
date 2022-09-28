import pandas as pd
from datetime import datetime
from pytz import timezone
from functools import reduce




class XPreprocessing:
    def __init__(self, start_utc:datetime, end_utc:datetime, local_tz:timezone):
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
        if data.empty:
            params = [self.local_time_col, 'load_mw']
            result = pd.DataFrame(columns=params)
            return result
        else:
            data.columns = data.columns.str.replace('HOUSTON', 'load_mw')
            data[self.utc_time_col] = pd.to_datetime(data[self.utc_time_col], utc=True)
            result = data.loc[(data[self.utc_time_col] >= self.start_utc) & (data[self.utc_time_col] <= self.end_utc), :]
            result = self._make_local_time(result)
        return result

    def preprocess_weather_historical(self, data: pd.DataFrame) -> pd.DataFrame:
        if data.empty:
            params = [self.local_time_col, 'temperature', 'precip', 'dew_point', 'humidity', 'wind', 'pressure']
            result = pd.DataFrame(columns=params)
            return result
        else:
            data.columns = data.columns.str.replace('time', self.utc_time_col)
            data[self.utc_time_col] = pd.to_datetime(data[self.utc_time_col], utc=True)

            data.insert(0, 'date_temp', data[self.utc_time_col].apply(lambda x: x.round(freq='H')))
            data = data.groupby(['date_temp']).first().reset_index()
            result = data.drop([self.utc_time_col], axis=1).rename(columns={'date_temp': self.utc_time_col})
            result = self._make_local_time(result)
            return result

    def preprocess_day_ahead(self, data: pd.DataFrame) -> pd.DataFrame:
        if data.empty:
            params = [self.local_time_col, 'day_ahead_price']
            result = pd.DataFrame(columns=params)
            return result
        else:
            data[self.utc_time_col] = pd.to_datetime(data[self.utc_time_col], utc=True)
            result = self._make_local_time(data)
            return result

    def preprocess_ancillary_mcpc(self, data: pd.DataFrame) -> pd.DataFrame:
        if data.empty:
            params = [self.local_time_col, 'mcpc_non_spin', 'mcpc_reg_down', 'mcpc_reg_up', 'mcpc_rrs']
            result = pd.DataFrame(columns=params)
            return result
        else:
            prefix='mcpc'
            anci_type_dict = {'Non Spin': f'{prefix}_non_spin',
                              'Regulation Down': f'{prefix}_regulation_down',
                              'Regulation Up': f'{prefix}_regulation_up',
                              'Responsive Reserve Service': f'{prefix}_responsive_reserve_service'}
            data['ancillarytype'] = data['ancillarytype'].apply(lambda x: anci_type_dict[x])
            result = pd.pivot_table(data, values='mcpc', index='date', columns='ancillarytype')
            result = result.reset_index()
            result[self.utc_time_col] = pd.to_datetime(result[self.utc_time_col], utc=True)
            result = self._make_local_time(result)
            return result

    def merge_x(self, x_data_list: list) -> pd.DataFrame:
        result = reduce(lambda left, right: pd.merge(left, right, on=self.local_time_col,
                                                     how='outer'), x_data_list)
        return result

