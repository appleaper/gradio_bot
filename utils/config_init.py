import os
import torch
from config import conf_yaml
from utils.tool import save_json_file, read_json_file, get_ollama_model_list


project_dir = os.path.dirname(os.path.dirname(__file__))

'''模型地址'''
model_dir = os.path.join(project_dir, 'model')
qwen25_05B_Instruct_model_path = os.path.join(model_dir, 'Qwen25_05B_Instruct')
deep_seek_r1_15b_model_path = os.path.join(model_dir, 'deepseek_r1_distill_qwen_1_5b')
StepfunOcr_model_path = os.path.join(model_dir, 'stepfun-aiGOT-OCR2_0')
multimodal_model_path = os.path.join(model_dir, 'openbmbMiniCPM-V-2_6-int4')
bge_m3_model_path = os.path.join(model_dir, 'BAAIbge-m3')
voice_model_path = os.path.join(model_dir, 'FireRedASR-AED-L')
chat_model_dict = {
    'qwen2.5':['0.5B-Instruct'],
    'deepseek':['1.5B']
}

# todo：这列不太智能，只能暂时手动的添加，看看怎么改一下
name2path = {
    'qwen2.5-0.5B-Instruct': qwen25_05B_Instruct_model_path,
    'StepfunOcr':StepfunOcr_model_path,
    'deepseek-1.5B':deep_seek_r1_15b_model_path,
    'FireRedAsr':voice_model_path
}

'''ollama配置'''
ollama_support_list = ['ollama-qwen2.5:0.5b']
chat_model_dict['ollama'] = ['qwen2.5:0.5b']

'''cuda设备选择'''
if torch.cuda.is_available():
    device = torch.device('cuda')
    device_str = 'cuda'
else:
    device = torch.device('cpu')
    device_str = 'cpu'

'''聊天相关'''
qwen_support_list = conf_yaml['local_chat']['qwen_support']
max_rag_len = conf_yaml['rag']['max_rag_len']   # rag的最大长度
max_history_len = conf_yaml['local_chat']['max_history_len']       # 聊天的记录的最大历史记录数
rag_top_k = conf_yaml['rag']['top_k']       # rag检索时返回多少条相关内容

'''中间数据'''
data_dir = os.path.join(project_dir, 'data')
rag_data_csv_dir = os.path.join(data_dir, 'rag', 'data_csv')
tmp_dir_path = os.path.join(data_dir, 'tmp')
font_path = os.path.join(data_dir, 'ocr', 'font', 'SimHei.ttf')

'''数据库相关'''
database_dir = os.path.join(data_dir, 'database')
database_type = conf_yaml['rag']['database']['choise']

'''lancedb配置相关'''
lancedb_data_dir = os.path.join(database_dir, 'lancedb')
lancedb_articles_user_filename = conf_yaml['rag']['database']['lancedb']['save']['articles_user_path']
lancedb_kb_article_map_filename = conf_yaml['rag']['database']['lancedb']['save']['kb_article_map_path']
lancedb_articles_user_path = os.path.join(lancedb_data_dir, lancedb_articles_user_filename)
lancedb_kb_article_map_path = os.path.join(lancedb_data_dir, lancedb_kb_article_map_filename)

'''milvus配置相关'''
milvus_data_dir = os.path.join(database_dir, 'milvus')
milvus_articles_user_filename = conf_yaml['rag']['database']['milvus']['save']['articles_user_path']
milvus_kb_article_map_filename = conf_yaml['rag']['database']['milvus']['save']['kb_article_map_path']
milvus_articles_user_path = os.path.join(milvus_data_dir, milvus_articles_user_filename)
milvus_kb_article_map_path = os.path.join(milvus_data_dir, milvus_kb_article_map_filename)

'''用户配置'''
config_dir = os.path.join(project_dir, 'config')
user_password_info_dict_filename = conf_yaml['user']['user_password_info_dict_path']
user_password_info_dict_path = os.path.join(config_dir, user_password_info_dict_filename)

def get_database_config():
    '''根据数据库进行初始化配置'''
    if database_type == 'lancedb':
        database_dir = os.path.join(lancedb_data_dir, 'data')
        articles_user_path = lancedb_articles_user_path
        kb_article_map_path = lancedb_kb_article_map_path
    elif database_type == 'milvus':
        database_dir = ''
        articles_user_path = milvus_articles_user_path
        kb_article_map_path = milvus_kb_article_map_path
    else:
        assert False, f"{database_type} not support!"
    return database_dir, articles_user_path, kb_article_map_path

'''parse voice config'''
voice_chunk_size = conf_yaml['rag']['parse_voice']['voice_chunk_size']



'''mysql配置'''
mysql_host = conf_yaml['mysql']['host']
mysql_port = conf_yaml['mysql']['port']
mysql_user = conf_yaml['mysql']['user']
mysql_password = conf_yaml['mysql']['password']
mysql_database_name = conf_yaml['mysql']['database_name']
mysql_article_table_info_name = conf_yaml['mysql']['article_table_name']

database_dir, articles_user_path, kb_article_map_path = get_database_config()

def init_article_user_and_kb_mapping_file(articles_user_path, kb_article_map_path):
    user_info_dict = read_json_file(user_password_info_dict_path)
    if os.path.exists(articles_user_path):
        pass
    else:
        articles_user_default_json = {}
        for user_name in user_info_dict.keys():
            articles_user_default_json[user_name] = {}
        save_json_file(articles_user_default_json, articles_user_path)

    if os.path.exists(kb_article_map_path):
        pass
    else:
        kb_article_map_default_json = {}
        for user_name in user_info_dict.keys():
            kb_article_map_default_json[user_name] = {}
        save_json_file(kb_article_map_default_json, kb_article_map_path)

init_article_user_and_kb_mapping_file(articles_user_path, kb_article_map_path)