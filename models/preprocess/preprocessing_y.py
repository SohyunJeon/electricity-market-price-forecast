import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pytz import utc


class CreateDate:
    def __init__(self, time_col_name: str, local_timezone=None):
        self.time_col = time_col_name
        self.local_time_col = f'local_{self.time_col}'
        self.local_tz = local_timezone

    def fit(self, data: pd.DataFrame):
        return self

    def transform(self, data:pd.DataFrame, tag:str='train') -> pd.DataFrame:
        df = data.copy()
        df = self.change_utc_to_local(df)
        if tag == 'train':
            df = self.get_day_name(df)
        if tag == 'inference':
            df = self.get_ahead_day_name(df)

        return df

    def change_utc_to_local(self, data):
        data[self.time_col] = pd.to_datetime(data[self.time_col], utc=True)
        # data[self.local_time_col] = data[self.time_col].apply(lambda x: utc.localize(x).astimezone(self.local_tz))
        data[self.local_time_col] = data[self.time_col].apply(lambda x: x.astimezone(self.local_tz))
        return data


    def get_day_name(self, data: pd.DataFrame, date_col: str=None) -> pd.DataFrame:
        if date_col == None:
            date_col = self.local_time_col
        data['week'] = data[date_col].dt.day_name()
        return data

    def get_ahead_day_name(self, data: pd.DataFrame, date_col: str=None) -> pd.DataFrame:
        if date_col == None:
            date_col = self.local_time_col
        data['week'] = (data[date_col] + timedelta(days=1)).dt.day_name()
        return data



class SummaryData:
    def __init__(self, y_col: str, time_col_name:str, pre_date_cnt:int=7, dummy_cols:list=['week']):
        self.y_col = y_col
        self.time_col = time_col_name
        self.pre_date_cnt = pre_date_cnt
        self.dummy_cols = dummy_cols
        self.dummy_val_list = []

    def fit(self, data: pd.DataFrame):
        return self

    def transform(self, data: pd.DataFrame, use_summ: bool=True, tag: str='train', inference_date: datetime=None) -> pd.DataFrame:
        df = data.copy()
        if tag == 'train':
            df = self.create_train_time_features(df, use_summ)
            df = self.create_train_dummy(df)
            df_id = data.loc[:, ['master_id', 'local_date']]
            df = pd.merge(df_id, df, on='local_date')
        if tag == 'inference':
            # df = self.create_train_time_features(df, use_summ)
            # df = self.create_train_dummy(df)
            df = self.create_inference_time_features(df, use_summ, inference_date)
            df = self.create_inference_dummy(df)
        return df


    def create_train_time_features(self, data: pd.DataFrame, use_summ) -> pd.DataFrame:
        result = pd.DataFrame()
        min_date = data[self.time_col].min()

        for i, row in data.iterrows():
            feat_start = row[self.time_col] - timedelta(days=self.pre_date_cnt)
            if feat_start < min_date:
                continue
            y = row[self.y_col]

            feat_dates = [row[self.time_col] - timedelta(days=day) for day in range(1, 8)]
            n_day_before = []
            for feat_date in feat_dates:
                try:
                    n_day_before.append(data.loc[data[self.time_col] == feat_date, self.y_col].values[0])
                except:
                    n_day_before.append(None)

            pre_date = row[self.time_col].date() - timedelta(days=1)
            pre_date_values = data.loc[data[self.time_col].dt.date == pre_date, self.y_col].values
            if use_summ:
                pre_date_feat = [pre_date_values.min(), pre_date_values.max(), pre_date_values.mean()]
            else:
                pre_date_feat = pre_date_values.tolist()

            new_feats = pd.DataFrame([row[self.time_col], row['week'], y,
                                      *n_day_before, *pre_date_feat]).T
            result = pd.concat([result, new_feats])

        n_day_before_name = ['D_'+str(idx) for idx in np.arange(1, (self.pre_date_cnt+1))]
        if use_summ:
            pre_date_feat_name = ['day_before_min', 'day_before_max', 'day_before_avg']
        else:
            pre_date_feat_name = ['X_'+str(i) for i in np.arange(1, 25)]
        result.columns = [self.time_col, 'week', 'y', *n_day_before_name, *pre_date_feat_name]
        result.reset_index(drop=True, inplace=True)
        return result

    def create_inference_time_features(self, data: pd.DataFrame, use_summ, inference_date) -> pd.DataFrame:
        result = pd.DataFrame()
        min_date = data[self.time_col].min()
        # max_date = data[self.time_col].max().date()
        max_date = (inference_date - timedelta(days=1)).date()
        feat_df = data.loc[data[self.time_col].dt.date==max_date, :].reset_index(drop=True)

        for i, row in feat_df.iterrows():
            y = row[self.y_col]

            feat_dates = [row[self.time_col] - timedelta(days=day) for day in range(0, self.pre_date_cnt)]
            n_day_before = []
            for feat_date in feat_dates:
                try:
                    n_day_before.append(data.loc[data[self.time_col]==feat_date, self.y_col].values[0])
                except:
                    n_day_before.append(None)
            # n_day_before = data.loc[data[self.time_col].isin(feat_dates), 'price'].values.tolist()[::-1]

            pre_date = row[self.time_col].date()
            pre_date_values = data.loc[data[self.time_col].dt.date == pre_date, self.y_col].values
            if use_summ:
                pre_date_feat = [pre_date_values.min(), pre_date_values.max(), pre_date_values.mean()]
            else:
                pre_date_feat = pre_date_values.tolist()

            new_feats = pd.DataFrame([row[self.time_col], row['week'], y,
                                      *n_day_before, *pre_date_feat]).T
            result = pd.concat([result, new_feats])

        n_day_before_name = ['D_' + str(idx) for idx in np.arange(1, (self.pre_date_cnt + 1))]
        if use_summ:
            pre_date_feat_name = ['day_before_min', 'day_before_max', 'day_before_avg']
        else:
            pre_date_feat_name = ['X_' + str(i) for i in np.arange(1, 25)]
        result.columns = [self.time_col, 'week', 'y', *n_day_before_name, *pre_date_feat_name]
        result.reset_index(drop=True, inplace=True)
        return result



    def create_inference_time_features_old(self, data: pd.DataFrame, use_summ) -> pd.DataFrame:
        result = pd.DataFrame()
        for i, row in data.iterrows():
            if i < ((self.pre_date_cnt-1) * 24):
                continue
            y = row[self.y_col]

            n_day_before_i = i - 24 * np.arange(0, (self.pre_date_cnt))
            n_day_before = data.loc[n_day_before_i, self.y_col].astype(float)

            pre_date = row[self.time_col].date()
            pre_date_values = data.loc[data[self.time_col].dt.date == pre_date, self.y_col].values
            if use_summ:
                pre_date_feat = [pre_date_values.min(), pre_date_values.max(), pre_date_values.mean()]
            else:
                pre_date_feat = pre_date_values.tolist()
            new_feats = pd.DataFrame([row[self.time_col], row['week'], y,
                                      *n_day_before.values, *pre_date_feat]).T
            result = pd.concat([result, new_feats])

        n_day_before_name = ['D_'+str(idx) for idx in np.arange(1, (self.pre_date_cnt+1))]
        if use_summ:
            pre_date_feat_name = ['day_before_min', 'day_before_max', 'day_before_avg']
        else:
            pre_date_feat_name = ['X_' + str(i) for i in np.arange(1, 25)]
        result.columns = [self.time_col, 'week', 'y', *n_day_before_name, *pre_date_feat_name]
        result.reset_index(drop=True, inplace=True)
        return result


    def create_train_dummy(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        for dummy_col in self.dummy_cols:
            dummy_df = pd.get_dummies(df[dummy_col], prefix=dummy_col)
            df = pd.concat([df, dummy_df], axis=1)
            self.dummy_val_list.append(dummy_df.columns.tolist())

        return df

    def set_dummy_val_list(self, dummy_list: list):
        self.dummy_val_list.append(dummy_list)


    def create_inference_dummy(self, data: pd.DataFrame) -> pd.DataFrame:
        for i, dummy_col in enumerate(self.dummy_cols):
            dummy_df = pd.DataFrame({k: v for k, v in zip(self.dummy_val_list[i],
                                                         [0] * len(self.dummy_val_list[i]))},
                                    index=data.index)
            dummy_val = data[f'{dummy_col}']
            for idx, val in dummy_val.items():
                dummy_df.loc[idx, f'{dummy_col}_{val}'] = 1
            data = pd.concat([data, dummy_df], axis=1)
        return data