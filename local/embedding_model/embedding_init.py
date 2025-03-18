
from database_data.emb_model.init_model_bge_m3 import model_init as bge_m3_model_init
from utils.config_init import bge_model_path


def load_rag_model(model_name):
    if model_name == 'bge_m3':
        model, tokenizer = bge_m3_model_init(bge_model_path)
        return model, tokenizer
    else:
        assert False, 'rag model not support!'