import os

from config import conf_yaml
from utils.tool import save_json_file, read_json_file, get_ollama_model_list


project_dir = os.path.dirname(os.path.dirname(__file__))

'''模型地址'''
model_dir = os.path.join(project_dir, 'model')
qwen25_05B_Instruct_model_path = os.path.join(model_dir, 'Qwen25_05B_Instruct')
StepfunOcr_model_path = os.path.join(model_dir, 'stepfun-aiGOT-OCR2_0')
multimodal_model_path = os.path.join(model_dir, 'openbmbMiniCPM-V-2_6-int4')
bge_m3_model_path = os.path.join(model_dir, 'BAAIbge-m3')
voice_model_path = os.path.join(model_dir, 'FireRedASR-AED-L')
chat_model_dict = {'qwen2.5':['0.5B-Instruct']}
name2path = {
    "qwen2.5-0.5B-Instruct": qwen25_05B_Instruct_model_path,
}

ollama_support_list = ['ollama-qwen2.5:0.5b']

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


'''ollama配置'''
# ollama_model_list = get_ollama_model_list()
chat_model_dict['ollama'] = ['qwen2.5:0.5b']

'''video配置'''
video_mark_dir = os.path.join(data_dir, 'video_mark')
translation_dir = os.path.join(video_mark_dir, 'translation')
video_cut_save_dir = os.path.join(video_mark_dir, 'video_cut')
video_cut_record_path = os.path.join(video_cut_save_dir, 'video_cut_record.txt')
video_mark_csv_path = os.path.join(video_mark_dir, 'adult_video', 'movie_mark.csv')
video_translation_title_csv_path = os.path.join(translation_dir, 'translation.csv')
video_need_translation_title_path = os.path.join(translation_dir, 'need_translation.txt')

qwen_support_list = conf_yaml['local_chat']['qwen_support']
max_rag_len = conf_yaml['rag']['max_rag_len']
max_history_len = conf_yaml['local_chat']['max_history_len']
rag_top_k = conf_yaml['rag']['top_k']

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