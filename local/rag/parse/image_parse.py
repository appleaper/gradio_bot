import pandas as pd
from config import conf_yaml
from local.rag.rag_model import load_model_cached, load_bge_model_cached

rag_config = conf_yaml['rag']
bge_model_path = rag_config['beg_model_path']
rag_ocr_model_path = rag_config['ocr_model_path']

rag_list_config_path = rag_config['rag_list_config_path']


def parse_image_do(file_name):
    info_list = []
    model, tokenizer = load_model_cached(rag_ocr_model_path)
    model_bge = load_bge_model_cached(bge_model_path)
    ocr_result = model.chat(tokenizer, file_name, ocr_type='format')
    info = {}
    info['title'] = ''
    info['content'] = ocr_result
    info['page_count'] = ''
    info['vector'] = model_bge.encode(ocr_result, batch_size=1, max_length=8192)['dense_vecs'].tolist()
    info['file_from'] = file_name
    info_list.append(info)
    df = pd.DataFrame(info_list)
    return df