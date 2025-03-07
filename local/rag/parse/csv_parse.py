import os
import gradio as gr
import pandas as pd
from tqdm import  tqdm
from config import conf_yaml
from local.rag.rag_model import load_bge_model_cached

rag_config = conf_yaml['rag']
bge_model_path = rag_config['beg_model_path']
rag_data_csv_dir = rag_config['rag_data_csv_dir']
rag_list_config_path = rag_config['rag_list_config_path']

def parse_csv_do(csv_path):
    try:
        df = pd.read_csv(csv_path, encoding='utf8')
    except:
        gr.Warning('文件读取失败')
        df = pd.DataFrame()
    csv_file_name = os.path.basename(csv_path)
    model_bge = load_bge_model_cached(bge_model_path)
    result_list = []
    for index, row in tqdm(df.iterrows(), total=len(df)):
        info = {}
        input_str = f'标题/提问：{row.title}\n正文/回答:{row.content}\n'
        info['title'] = row['title']
        info['content'] = row['content']
        info['page_count'] = index+1
        info['vector'] = model_bge.encode(input_str, batch_size=1, max_length=8192)['dense_vecs'].tolist()
        info['file_from'] = csv_file_name
        result_list.append(info)
    result_df = pd.DataFrame(result_list)
    return result_df