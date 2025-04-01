import os
import hashlib
import gradio as gr
import pandas as pd
from tqdm import  tqdm
from local.rag.rag_model import load_bge_model_cached


def parse_csv_do(csv_path, id, user_id, database_type):
    file_name, suffix = os.path.splitext(os.path.basename(csv_path))
    if suffix == '.csv':
        try:
            df = pd.read_csv(csv_path, encoding='utf8')
        except:
            gr.Warning('文件读取失败')
            df = pd.DataFrame()
    elif suffix == '.xlsx':
        try:
            df = pd.read_excel(csv_path)
        except:
            gr.Warning('文件读取失败')
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()
        gr.Warning('数据类型不支持')

    if database_type in ['lancedb', 'milvus']:
        model_bge = load_bge_model_cached(bge_m3_model_path)
    result_list = []
    columns_list = df.columns
    if 'title' in columns_list and 'content' in columns_list:
        for index, row in tqdm(df.iterrows(), total=len(df)):
            info = {}
            input_str = f'标题/提问：{row.title}\n正文/回答:{row.content}\n'
            info['user_id'] = user_id
            info['article_id'] = id
            info['title'] = row['title']
            info['content'] = row['content']
            info['page_count'] = str(index+1)
            if database_type in ['lancedb', 'milvus']:
                info['vector'] = model_bge.encode(input_str, batch_size=1, max_length=8192)['dense_vecs'].tolist()
            else:
                info['vector'] = []
            info['file_from'] = file_name
            info['hash_check'] = hashlib.sha256((user_id+id+row['title']+row['content']).encode('utf-8')).hexdigest()
            result_list.append(info)
        result_df = pd.DataFrame(result_list)
        return result_df
    else:
        gr.Warning(f'{file_name} 应该含"title"列和"content"列，代码只解析这两列')
        return pd.DataFrame()
