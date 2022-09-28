from qpslib_model_manager import score_info
import dill


def save_model_score(client, model_id, score):
    score_obj = score_info.ScoreInfo(model_id, score)
    score_resp = client.save_score([score_obj])
    print('Save score finish.')


def save_sub_models(client, sub_models, name: str, desc: str='', meta: dict={}, feats: dict={}):
    sub_models_b = dill.dumps(sub_models)
    sub_resp = client.upload_sub_model(name=name,
                                       model=sub_models_b,
                                       description=desc,
                                       feature_importance=feats,
                                       meta=meta)
    return sub_resp


def save_stacked_model(client, stacked_model, ref_id, score, name: str, desc:str='', meta:dict={}, feats: dict={}):
    stacked_model_b = dill.dumps(stacked_model)
    stacked_resp = client.upload_stacked_model(ref=ref_id,
                                               name=name,
                                               model=stacked_model_b,
                                               description=desc,
                                               feature_importance=feats,
                                               meta=meta)
    print(f'Uploaded Stacked-model id : {stacked_resp.id}')
    save_model_score(client, stacked_resp.id, score)
    return stacked_resp


