from pymongo import MongoClient
import pymssql
import pymysql
import yaml


class DBConnection:
    def __init__(self, host: str, port: int, username: str, password: str, database: str):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.database = database

    def get_db_conn(self):
        pass


class MongoDBConnection(DBConnection):
    def __init__(self, host: str, port: int, username: str, password: str, database: str):
        super().__init__(host, port, username, password, database)


    def get_db_conn(self) -> MongoClient:
        conn = MongoClient(host=self.host,
                           port=self.port,
                           username=self.username,
                           password=self.password,
                           authSource=self.database)
        db = conn[self.database]
        return conn, db


class MariaDBConnection(DBConnection):
    def __init__(self, host: str, port: int, username: str, password: str, database: str):
        super().__init__(host, port, username, password, database)

    def get_db_conn(self) -> pymysql:
        conn = pymysql.connect(
            user=self.username,
            password=self.password,
            host=self.host,
            database=self.database,
            port=self.port
        )
        return conn




class MSSQLConnection(DBConnection):
    def __init__(self, host: str, port: int, username: str, password: str, database: str):
        super().__init__(host, port, username, password, database)

    def get_db_conn(self) -> pymssql:
        conn = pymssql.connect(
            host=self.host,
            port=self.port,
            user=self.username,
            password=self.password,
            database=self.database
        )
        return conn


if __name__  ==  '__main__':
    # get db info
    with open('./config.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    mongo_info = config['mongo_info']
    maria_info = config['maria_info']

    # get mongodb connection
    mongo_conn, mongo_db = MongoDBConnection(host=mongo_info['host'],
                                             port=mongo_info['port'],
                                             username=mongo_info['username'],
                                             password=mongo_info['password'],
                                             database=mongo_info['database']).get_db_conn()
    print(mongo_db)

    # get maria connection
    maria_conn = MariaDBConnection(host=maria_info['host'],
                                 port=maria_info['port'],
                                 username=maria_info['username'],
                                 password=maria_info['password'],
                                 database=maria_info['database']).get_db_conn()
    print(maria_conn)