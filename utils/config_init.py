import os
from utils.tool import save_json_file, read_json_file
from config import conf_yaml

bge_model_path = conf_yaml['rag']['embedding_mdoel']['bge_m3']['model_path']
qwen_support_list = conf_yaml['local_chat']['qwen_support']
max_rag_len = conf_yaml['rag']['max_rag_len']
max_history_len = conf_yaml['local_chat']['max_history_len']
rag_top_k = conf_yaml['rag']['top_k']

rag_ocr_model_path = conf_yaml['rag']['parse_image']['model_path']
rag_data_csv_dir = conf_yaml['rag']['rag_data_csv_dir']
trie_dir_path  = conf_yaml['rag']['tokenizer']['trie_dir_path']
tmp_dir_path = conf_yaml['rag']['tmp_dir_path']
voice_model_path = conf_yaml['rag']['parse_voice']['model_path']
voice_chunk_size = conf_yaml['rag']['parse_voice']['voice_chunk_size']
user_password_info_dict_path = conf_yaml['user']['user_password_info_dict_path']

lancedb_articles_user_path = conf_yaml['rag']['database']['lancedb']['save']['articles_user_path']
lancedb_kb_article_map_path = conf_yaml['rag']['database']['lancedb']['save']['kb_article_map_path']
milvus_articles_user_path = conf_yaml['rag']['database']['milvus']['save']['articles_user_path']
milvus_kb_article_map_path = conf_yaml['rag']['database']['milvus']['save']['kb_article_map_path']
database_type = conf_yaml['rag']['database']['choise']
def get_database_config():
    '''根据数据库进行初始化配置'''
    if database_type == 'lancedb':
        database_dir = conf_yaml['rag']['database']['lancedb']['save']['database_dir']
        articles_user_path = lancedb_articles_user_path
        kb_article_map_path = lancedb_kb_article_map_path
    elif database_type == 'milvus':
        database_dir = ''
        articles_user_path = milvus_articles_user_path
        kb_article_map_path = milvus_kb_article_map_path
    else:
        assert False, f"{database_type} not support!"
    return database_dir, articles_user_path, kb_article_map_path

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