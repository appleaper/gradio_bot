import os
import hashlib
import pandas as pd
from tqdm import tqdm
from local.rag.rag_model import load_bge_model_cached
from local.rag.util import split_by_heading
from utils.tool import read_md_doc
from utils.tool import hash_code, slice_string, chunk_str

def parse_markdown_do(file_path, user_id, database_type, emb_model_class):
    markdown_data = read_md_doc(file_path)
    emb_model_name = emb_model_class.model_name
    markdown_data_list = split_by_heading(markdown_data, level=2)
    file_name, _ = os.path.splitext(os.path.basename(file_path))
    article_id = hash_code(file_name)
    result_list = []
    for info_i in markdown_data_list:
        text_list = slice_string(info_i, punctuation=r'[，。！？,.]')
        for content in text_list:
            info = chunk_str('', content, user_id, article_id, '', emb_model_name, database_type, file_name,
                             emb_model_class)
            result_list.append(info)
    result_df = pd.DataFrame(result_list)
    return result_df