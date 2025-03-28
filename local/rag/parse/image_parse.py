import hashlib
import pandas as pd
from local.rag.rag_model import load_model_cached, load_bge_model_cached

from utils.config_init import bge_m3_model_path, StepfunOcr_model_path

def parse_image_do(file_name, id, user_id, database_type):
    info_list = []
    model, tokenizer = load_model_cached(StepfunOcr_model_path)
    if database_type in ['lancedb', 'milvus']:
        model_bge = load_bge_model_cached(bge_m3_model_path)
    ocr_result = model.chat(tokenizer, file_name, ocr_type='format')
    info = {}
    info['user_id'] = user_id
    info['article_id'] = id
    info['title'] = ''
    info['content'] = ocr_result
    info['page_count'] = ''
    if database_type in ['lancedb', 'milvus']:
        info['vector'] = model_bge.encode(ocr_result, batch_size=1, max_length=8192)['dense_vecs'].tolist()
    else:
        info['vector'] = []
    info['file_from'] = file_name
    info['hash_check'] = hashlib.sha256((user_id+id+ocr_result).encode('utf-8')).hexdigest()
    info_list.append(info)
    df = pd.DataFrame(info_list)
    return df