from pymongo.collection import Collection
import pandas as pd
from datetime import datetime

def load_features_with_duration(collection: Collection, dt_param_name: str,
                                search_start: str, search_end: str) -> pd.DataFrame:
    docs = collection.find({dt_param_name: {'$gte': search_start, '$lt': search_end}},
                           {'_id': 0}).sort(dt_param_name, -1)
    result = pd.DataFrame(docs)
    result = result.sort_values(dt_param_name).reset_index(drop=True)
    result[dt_param_name] = result[dt_param_name].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    return result


def load_features_with_count(collection: Collection, dt_param_name: str, feats_cnt: int) -> pd.DataFrame:
    docs = collection.find({}, {'_id': 0}).sort(dt_param_name, -1).limit(feats_cnt)
    result = pd.DataFrame(docs)
    result = result.sort_values(dt_param_name).reset_index(drop=True)
    result[dt_param_name] = result[dt_param_name].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    return result


def save_features_to_db(collection: Collection, data: pd.DataFrame):
    for label, content in data.iterrows():
        data_dict = dict(content)
        resp = collection.update_one({'master_id': content['master_id']},
                               {'$set': data_dict}, upsert=True)
        print(f'{label}:{resp.raw_result}')
