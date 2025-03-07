import lancedb
import os
import uuid
import json
import gradio as gr
from tqdm import tqdm
import pandas as pd
from config import conf_yaml
from local.rag.parse.pdf_parse import parse_pdf_do
from local.rag.parse.csv_parse import parse_csv_do
from local.rag.parse.markdown_parse import parse_markdown_do
from local.rag.parse.docx_parser import parse_docx_do
from local.rag.parse.image_parse import deal_images_group
from local.rag.util import read_rag_name_dict

rag_config = conf_yaml['rag']
rag_database_name = rag_config['rag_database_name']
rag_list_config_path = rag_config['rag_list_config_path']
rag_data_csv_dir = rag_config['rag_data_csv_dir']


def get_rag_now_dict(data):
    '''id和文章名字相反'''
    inverted_dict = {value: key for key, value in data.items()}
    return inverted_dict

def save_rag_group_name(group_name, history_rag_dict, rag_list_config_path):
    '''给组名返回一个唯一的id'''
    if group_name not in history_rag_dict.values():
        id = str(uuid.uuid4())[:8]
        history_rag_dict[id] = group_name
    else:
        id_list = [key for key, value in history_rag_dict.items() if value == group_name]
        id = id_list[0]
    with open(rag_list_config_path, 'w', encoding='utf8') as json_file:
        json.dump(history_rag_dict, json_file, indent=4, ensure_ascii=False)  # 使用indent参数美化输出
    return id

def save_rag_group_csv_name(df2, rag_data_csv_dir, id, rag_list_config_path):
    history_rag_dict = read_rag_name_dict(rag_list_config_path)
    csv_name = history_rag_dict[id]
    save_path = os.path.join(rag_data_csv_dir, csv_name + '.csv')
    if os.path.exists(save_path):
        df1 = pd.read_csv(save_path, encoding='utf8')
        df2 = pd.concat([df1, df2], ignore_index=True)
        df2 = df2.drop_duplicates(subset=['title','content','page_count','file_from'])
        df2.to_csv(save_path, index=False, encoding='utf8')
    else:
        df2.to_csv(save_path, index=False, encoding='utf8')
    return save_path

def create_or_add_data_to_lancedb(rag_database_name, table_name, df):
    db = lancedb.connect(rag_database_name)
    table_path = os.path.join(rag_database_name, table_name + '.lance')
    if not os.path.exists(table_path):
        tb = db.create_table(table_name, data=df, exist_ok=True)
        row_count = tb.count_rows()
        gr.Info(f'创建新的数据表，并写入{row_count}条数据')
    else:
        tb = db.open_table(table_name)
        old_count = tb.count_rows()
        tb.add(data=df)
        now_count = tb.count_rows()
        gr.Info(f'写入{now_count - old_count}条记录, 现有{now_count}条记录')
def deal_mang_knowledge_files(rag_upload_files, is_same_group, knowledge_name, progress=gr.Progress()):
    if knowledge_name == '':
        knowledge_name = os.path.basename(rag_upload_files[0])
    data = read_rag_name_dict(rag_list_config_path)
    id = save_rag_group_name(knowledge_name, data, rag_list_config_path)
    for file_index, file_name in tqdm(enumerate(rag_upload_files), total=len(rag_upload_files)):
        upload_file, suffix = os.path.splitext(os.path.basename(file_name))
        if suffix == '.pdf':
            df = parse_pdf_do(file_name)
        elif suffix == '.csv':
            df = parse_csv_do(file_name)
        elif suffix == '.md':
            df = parse_markdown_do(file_name)
        elif suffix == '.docx':
            df = parse_docx_do(file_name)
        elif suffix in ['.jpg', '.jpeg', '.png']:
            df = deal_images_group(file_name)
        else:
            gr.Warning(f'{os.path.basename(file_name)}不支持解析')
            continue
        if is_same_group == '否' and file_index!=0:
            data = read_rag_name_dict(rag_list_config_path)
            id = save_rag_group_name(os.path.basename(file_name), data, rag_list_config_path)
        create_or_add_data_to_lancedb(rag_database_name, id, df)
        save_rag_group_csv_name(df, rag_data_csv_dir, id, rag_list_config_path)
        progress(round((file_index + 1) / len(rag_upload_files), 2))
    return get_rag_now_dict(data), None, None, []