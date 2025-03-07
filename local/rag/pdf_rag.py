import copy
import os
import lancedb
import gradio as gr

from local.rag.rag_model import load_bge_model_cached
from local.rag.util import read_rag_name_dict, write_rag_name_dict
from utils.tool import read_json_file, save_json_file
from config import conf_yaml
rag_config = conf_yaml['rag']
rag_list_config_path = rag_config['rag_list_config_path']
rag_data_csv_dir = rag_config['rag_data_csv_dir']
bge_model_path = rag_config['beg_model_path']
rag_database_name = rag_config['rag_database_name']
knowledge_base_info_save_path = rag_config['knowledge_base_info_save_path']

def search_similar(database_name, table_name, test_str, top_k):
    model_bge = load_bge_model_cached(bge_model_path)
    vector = model_bge.encode(test_str, batch_size=1, max_length=8192)['dense_vecs'].tolist()
    db = lancedb.connect(database_name)
    tb = db.open_table(table_name)
    records = tb.search(vector).limit(top_k).to_pandas()
    return records

def drop_lancedb_table(table_name_list):
    '''删除文章'''
    db = lancedb.connect(rag_database_name)
    data = read_rag_name_dict(rag_list_config_path)
    inverted_dict = {value: key for key, value in data.items()}

    knowledge_json = read_json_file(knowledge_base_info_save_path)
    for table_name in table_name_list:
        if table_name in inverted_dict:
            table_path = os.path.join(rag_database_name, inverted_dict[table_name] + '.lance')
            if os.path.exists(table_path):
                db.drop_table(inverted_dict[table_name])
                del data[inverted_dict[table_name]]
                del inverted_dict[table_name]
                gr.Info(f'成功删除{table_name}')
            else:
                del data[inverted_dict[table_name]]
                del inverted_dict[table_name]
                gr.Warning(f'数据不存在，但依旧执行删除')
            csv_drop_path = os.path.join(rag_data_csv_dir, table_name + '.csv')
            if os.path.exists(csv_drop_path):
                os.remove(csv_drop_path)

    knowledge_json_copy = copy.deepcopy(knowledge_json)
    for key in list(knowledge_json.keys()):
        value_list = knowledge_json[key]
        for element_to_remove in table_name_list:
            while element_to_remove in value_list:
                value_list.remove(element_to_remove)
        if not value_list:
            # 如果列表为空，删除该键值对
            del knowledge_json_copy[key]
    save_json_file(knowledge_json_copy, knowledge_base_info_save_path)
    write_rag_name_dict(rag_list_config_path, data)
    return inverted_dict, knowledge_json_copy

def select_lancedb_table():
    data = read_rag_name_dict(rag_list_config_path)
    inverted_dict = {value: key for key, value in data.items()}
    return gr.Dropdown(choices=list(inverted_dict.keys()), label="上下文知识")

if __name__ == '__main__':
    pass