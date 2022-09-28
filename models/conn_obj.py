
from common.connect_db import MariaDBConnection, MongoDBConnection
from models.conf import config


class DBConn():
    def __init__(self):

        self.dev_conn_weather = MariaDBConnection(host=config['esbr_dev_conn']['host'],
                                     port=config['esbr_dev_conn']['port'],
                                     username=config['esbr_dev_conn']['username'],
                                     password=config['esbr_dev_conn']['password'],
                                     database='wunderground').get_db_conn()

        self.dev_conn_ercot = MariaDBConnection(host=config['esbr_dev_conn']['host'],
                                     port=config['esbr_dev_conn']['port'],
                                     username=config['esbr_dev_conn']['username'],
                                     password=config['esbr_dev_conn']['password'],
                                     database='ercot').get_db_conn()

        self.dev_conn_eo = MariaDBConnection(host=config['esbr_dev_conn']['host'],
                                     port=config['esbr_dev_conn']['port'],
                                     username=config['esbr_dev_conn']['username'],
                                     password=config['esbr_dev_conn']['password'],
                                     database='energyonline').get_db_conn()


        self.esbr_summ_conn, self.esbr_summ_db = MongoDBConnection(host=config['esbr_conn']['host'],
                                     port=config['esbr_conn']['port'],
                                     username=config['esbr_conn']['username'],
                                     password=config['esbr_conn']['password'],
                                     database='summary').get_db_conn()

        self.esbr_raw_conn, self.esbr_raw_db = MongoDBConnection(host=config['esbr_conn']['host'],
                                     port=config['esbr_conn']['port'],
                                     username=config['esbr_conn']['username'],
                                     password=config['esbr_conn']['password'],
                                     database='eip').get_db_conn()


        self.esbr_history_conn, self.esbr_history_db = MongoDBConnection(host=config['esbr_conn']['host'],
                                     port=config['esbr_conn']['port'],
                                     username=config['esbr_conn']['username'],
                                     password=config['esbr_conn']['password'],
                                     database='history').get_db_conn()

        self.esbr_model_conn, self.esbr_model_db = MongoDBConnection(host=config['esbr_conn']['host'],
                                                                         port=config['esbr_conn']['port'],
                                                                         username=config['esbr_conn']['username'],
                                                                         password=config['esbr_conn']['password'],
                                                                         database='model').get_db_conn()