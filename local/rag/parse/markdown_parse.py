import os
import hashlib
import pandas as pd
from tqdm import tqdm
from local.rag.rag_model import load_bge_model_cached
from local.rag.util import split_by_heading
from utils.tool import read_md_doc
from utils.config_init import bge_m3_model_path

def parse_markdown_do(md_path, id, user_id, database_type):
    markdown_data = read_md_doc(md_path)
    markdown_data_list = split_by_heading(markdown_data, level=2)
    if database_type in ['lancedb', 'milvus']:
        model_bge = load_bge_model_cached(bge_m3_model_path)
    file_name = os.path.basename(md_path)
    result_list = []
    for index, markdown_line in tqdm(enumerate(markdown_data_list), total=len(markdown_data_list)):
        info = {}
        input_str = f'标题/提问：{markdown_line["title"]}\n正文/回答:{markdown_line["content"]}\n'
        info['user_id'] = user_id
        info['article_id'] = id
        info['title'] = markdown_line["title"]
        info['content'] = markdown_line["content"]
        info['page_count'] = ''
        if database_type in ['lancedb', 'milvus']:
            info['vector'] = model_bge.encode(input_str, batch_size=1, max_length=8192)['dense_vecs'].tolist()
        else:
            info['vector'] = []
        info['file_from'] = file_name
        info['hash_check'] = hashlib.sha256((user_id+id+markdown_line["title"]+markdown_line["content"]).encode('utf-8')).hexdigest()
        result_list.append(info)
    result_df = pd.DataFrame(result_list)
    return result_df