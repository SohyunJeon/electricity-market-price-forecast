

config = {
  'esbr_conn':{
    'host': '',
    'port': 27017,
    'username': '',
    'password': ''
  },
  'esbr_dev_conn':{
    'host': '',
    'port': 3306,
    'username': '',
    'password': ''
  },
  'timezone':
    'US/Central'

}

class ModelParams:
    local_date ='local_date'
    train_days = 30
    outlier = 100
    y_params = ['D_1', 'D_2', 'day_before_avg', 'day_before_min', 'day_before_max', ]
    load_params = ['load_mw']
    weather_params = ['temperature', 'precip', 'dew_point', 'humidity', 'wind', 'pressure']
    day_ahead_params = ['day_ahead_price']
    anci_params = ['mcpc_non_spin', 'mcpc_regulation_down', 'mcpc_regulation_up',
                   'mcpc_responsive_reserve_service']

    y_params_pow = [x + '_pow' for x in y_params]
    load_params_pow = [x + '_pow' for x in load_params]
    weather_params_pow = [x + '_pow' for x in weather_params]
    day_ahead_params_pow = [x + '_pow' for x in day_ahead_params]
    anci_params_pow = [x + '_pow' for x in anci_params]

    total_params = y_params + load_params + weather_params + day_ahead_params + anci_params
    total_pow_params = total_params + y_params_pow + load_params_pow + weather_params_pow + \
                       day_ahead_params_pow + anci_params_pow




class EvaluationItems:
    dayname_dict = {0: 'Sun', 1: 'Mon', 2: 'Tue', 3: 'Wed', 4: 'Thu', 5: 'Fri', 6: 'Sat'}
    month_dict = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul',
                  8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}