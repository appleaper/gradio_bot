import os
import uuid
import json
import lancedb
import pandas as pd
import gradio as gr
from tqdm import tqdm
from local.rag.util import save_info_to_lancedb, read_rag_name_dict
from local.rag.rag_model import load_model_cached, load_bge_model_cached

from config import conf_yaml
rag_config = conf_yaml['rag']
rag_top_k = rag_config['top_k']
rag_ocr_model_path = rag_config['ocr_model_path']
rag_data_csv_dir = rag_config['rag_data_csv_dir']
bge_model_path = rag_config['beg_model_path']
rag_database_name = rag_config['rag_database_name']
rag_list_config_path = rag_config['rag_list_config_path']

def save_rag_group_name(group_name, history_rag_dict, rag_list_config_path):
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

def get_rag_now_list():
    data = read_rag_name_dict(rag_list_config_path)
    inverted_dict = {value: key for key, value in data.items()}
    rag_deal_list = list(inverted_dict.keys())
    return rag_deal_list

def deal_images_group(group_name, rag_files, progress=gr.Progress()):

    info_list = []
    model, tokenizer = load_model_cached(rag_ocr_model_path)
    model_bge = load_bge_model_cached(bge_model_path)
    for index, file_name in tqdm(enumerate(rag_files), total=len(rag_files)):
        upload_file, suffix = os.path.splitext(os.path.basename(file_name))
        if suffix in ['.jpg', 'jpeg', 'png']:
            ocr_result = model.chat(tokenizer, file_name, ocr_type='format')
            info = {}
            info['title'] = ''
            info['content'] = ocr_result
            info['page_count'] = ''
            info['vector'] = model_bge.encode(ocr_result, batch_size=1, max_length=8192)['dense_vecs'].tolist()
            info['file_from'] = upload_file + suffix
            info_list.append(info)
        progress(round((index + 1) / len(rag_files), 2))
    df = pd.DataFrame(info_list)
    data = read_rag_name_dict(rag_list_config_path)
    id = save_rag_group_name(group_name, data, rag_list_config_path)
    df_save_path = save_rag_group_csv_name(df, rag_data_csv_dir, id, rag_list_config_path)
    return df, df_save_path, id

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

def new_files_rag(rag_files, group_name):
    if group_name == '':
        gr.Warning('请给群组起一个名字')
        rag_deal_list = get_rag_now_list()
        return gr.CheckboxGroup(choices=rag_deal_list, label="rag列表"), gr.Dropdown(choices=rag_deal_list, label="上下文知识")
    else:
        df, df_save_path, id = deal_images_group(group_name, rag_files)
        df = save_info_to_lancedb(df)
        create_or_add_data_to_lancedb(rag_database_name, id, df)
        rag_deal_list = get_rag_now_list()
        return gr.CheckboxGroup(choices=rag_deal_list, label="rag列表"), gr.Dropdown(choices=rag_deal_list,
                                                                                     label="上下文知识")