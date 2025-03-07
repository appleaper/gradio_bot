import os
import gradio as gr
import pandas as pd
from tqdm import tqdm
from config import conf_yaml
from local.rag.rag_model import load_bge_model_cached
from local.rag.util import read_md_doc, split_by_heading

rag_config = conf_yaml['rag']
bge_model_path = rag_config['beg_model_path']
rag_data_csv_dir = rag_config['rag_data_csv_dir']
rag_list_config_path = rag_config['rag_list_config_path']


def parse_markdown_do(md_path):
    markdown_data = read_md_doc(md_path)
    markdown_data_list = split_by_heading(markdown_data, level=2)
    model_bge = load_bge_model_cached(bge_model_path)
    file_name = os.path.basename(md_path)
    result_list = []
    for index, markdown_line in tqdm(enumerate(markdown_data_list), total=len(markdown_data_list)):
        info = {}
        input_str = f'标题/提问：{markdown_line["title"]}\n正文/回答:{markdown_line["content"]}\n'
        info['title'] = markdown_line["title"]
        info['content'] = markdown_line["content"]
        info['page_count'] = ''
        info['vector'] = model_bge.encode(input_str, batch_size=1, max_length=8192)['dense_vecs'].tolist()
        info['file_from'] = file_name
        result_list.append(info)
    result_df = pd.DataFrame(result_list)
    return result_df