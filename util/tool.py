import os
import json
import uuid
from datetime import datetime

import pandas as pd


def load_html_file(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        html_content = file.read()
    return html_content

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

def save_rag_name_dict(pdf_name, history_rag_dict, rag_list_config_path):
    now = datetime.now()
    date_time_str = now.strftime("%Y_%m_%d_%H_%M_%S")
    file_name, end_suff = os.path.splitext(os.path.basename(pdf_name))
    while True:
        id = str(uuid.uuid4())[:8]
        if id not in history_rag_dict.keys():
            if file_name not in history_rag_dict.values():
                history_rag_dict[id] = file_name
            else:
                file_name = file_name + '_' + date_time_str
                history_rag_dict[id] = file_name
            break

    with open(rag_list_config_path, 'w', encoding='utf8') as json_file:
        json.dump(history_rag_dict, json_file, indent=4, ensure_ascii=False)  # 使用indent参数美化输出
    return id

def save_rag_csv_name(df, rag_data_csv_dir, id, rag_list_config_path):
    history_rag_dict = read_rag_name_dict(rag_list_config_path)
    csv_name = history_rag_dict[id]
    save_path = os.path.join(rag_data_csv_dir, csv_name + '.csv')
    df.to_csv(save_path, index=False, encoding='utf8')
    return save_path

def write_rag_name_dict(path, rag_dict):
    with open(path, 'w', encoding='utf8') as json_file:
        json.dump(rag_dict, json_file, indent=4, ensure_ascii=False)  # 使用indent参数美化输出
def read_rag_name_dict(path):
    with open(path, 'r', encoding='utf8') as json_file:
        json_dict = json.load(json_file)
    return json_dict

def read_md_doc(path):
    with open(path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content
