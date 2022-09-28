import pandas as pd
from datetime import datetime, timedelta
from models.conn_obj import DBConn
from models.conf import ModelParams

conn = DBConn()

def load_data_from_db(name:str, start_utc:datetime, end_utc:datetime):
    """
    :param name: 모델 학습에 필요한 X데이터 이름
        day_ahead, weather, ancillary_services_mcp, load
    :param start_utc: 조회 시작 시간
    :param end_utc: 조회 종료 시간
    :return:
    """

    if name == 'day_ahead':
        day_ahead_query = f"""SELECT date, price AS day_ahead_price
                    FROM ercot_day_ahead_price
                    WHERE date >= '{start_utc}' AND date <= '{end_utc}' AND zone='LZ_HOUSTON'"""
        try:
            result = pd.read_sql(day_ahead_query, conn.dev_conn_eo)
        except:
            params = [ModelParams.local_date, 'temperature', 'precip', 'dew_point', 'humidity', 'wind', 'pressure']
            result = pd.DataFrame(columns=params)
        return result

    if name == 'weather':
        weather_feats = ['temperature', 'dew_point', 'humidity', 'wind_speed', 'pressure', 'precip']
        weather_feats_str = ','.join(['time'] + weather_feats)
        weather_feats_str = weather_feats_str.replace('wind_speed', 'wind_speed AS wind')

        weather_query = f""" SELECT {weather_feats_str} FROM houston_history_daily
                        WHERE time >= '{start_utc - timedelta(hours=1)}' AND time <= '{end_utc}'
                """
        try:
            result = pd.read_sql(weather_query, conn.dev_conn_weather)
        except:
            params = [ModelParams.local_date, 'temperature', 'precip', 'dew_point', 'humidity', 'wind', 'pressure']
            result = pd.DataFrame(columns=params)

        return result

    if name == 'ancillary_services_mcp':
        anci_query = f"""SELECT date, mcpc, ancillarytype
                      FROM ercot_ancillary_services_mcp
                      WHERE date >= '{start_utc}' AND date <= '{end_utc}'
                      """
        try:
            result = pd.read_sql(anci_query, conn.dev_conn_eo)
        except:
            params = [ModelParams.local_date, 'mcpc_non_spin', 'mcpc_reg_down', 'mcpc_reg_up', 'mcpc_rrs']
            result = pd.DataFrame(columns=params)
        return result

    if name=='load':
        load_query = f"""SELECT CONCAT(OperDay, ' ', HourEnding) AS date, HOUSTON FROM ercot_actual_load
                        WHERE CONCAT(OperDay, ' ', HourEnding) >= '{datetime.strftime(start_utc, '%Y-%m-%d %H:%M:%S')}' 
                            AND CONCAT(OperDay, ' ', HourEnding) <= '{datetime.strftime(end_utc, '%Y-%m-%d %H:%M:%S')}'
                        """
        try:
            result = pd.read_sql(load_query, conn.dev_conn_ercot)
        except:
            params = [ModelParams.local_date, 'load_mw']
            result = pd.DataFrame(columns=params)

        return result

