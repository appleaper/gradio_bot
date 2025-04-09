import os
import hashlib
import gradio as gr
import pandas as pd
from tqdm import  tqdm
from utils.tool import hash_code, slice_string, chunk_str

def parse_csv_do(file_path, user_id, database_type, embedding_class):
    emb_model_name = embedding_class.model_name
    file_name, suffix = os.path.splitext(os.path.basename(file_path))
    article_id = hash_code(file_name)
    if suffix == '.csv':
        try:
            df = pd.read_csv(file_path, encoding='utf8')
        except:
            gr.Warning('文件读取失败')
            df = pd.DataFrame()
    elif suffix == '.xlsx':
        try:
            df = pd.read_excel(file_path)
        except:
            gr.Warning('文件读取失败')
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()
        gr.Warning('数据类型不支持')

    result_list = []
    columns_list = df.columns
    if 'title' in columns_list and 'content' in columns_list:
        for index, row in tqdm(df.iterrows(), total=len(df)):
            if len(row['content'])+len(row['title']) > 0:
                text_list = slice_string(row['content'], punctuation=r'[，。！？,.]')
                for content in text_list:
                    title = row['title']
                    info = chunk_str(title, content, user_id, article_id, index, emb_model_name, database_type, file_name, embedding_class)
                    result_list.append(info)

        result_df = pd.DataFrame(result_list)
        return result_df
    else:
        gr.Warning(f'{file_name} 应该含"title"列和"content"列，代码只解析这两列')
        return pd.DataFrame()
