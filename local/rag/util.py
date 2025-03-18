import os
import re
import json
import uuid
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

def get_keys_from_value(dictionary, value):
    return [key for key, val in dictionary.items() if val == value]

def save_rag_name_dict(pdf_name, history_rag_dict, rag_list_config_path):
    now = datetime.now()
    date_time_str = now.strftime("%Y_%m_%d_%H_%M_%S")
    file_name, end_suff = os.path.splitext(os.path.basename(pdf_name))
    if file_name in list(history_rag_dict.values()):
        id = get_keys_from_value(history_rag_dict, file_name)[0]
    else:
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




def split_by_heading(text, level=2):
    # 使用正则表达式匹配二级标题
    pattern = r'(' +'#'*level +r' .+?)(?=\n##|\Z)'
    matches = re.findall(pattern, text, re.DOTALL)

    # 将匹配到的标题和内容分割成列表
    sections = []
    for match in matches:
        # 将标题和内容分开
        title, content = match.split('\n', 1)
        sections.append({'title': title.strip(), 'content': content.strip()})
    return sections