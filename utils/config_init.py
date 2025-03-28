import os
import torch
import gradio as gr
from config import conf_yaml
from utils.tool import read_user_info_dict, singleton
from utils.tool import save_json_file, read_json_file


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




'''用户配置'''
config_dir = os.path.join(project_dir, 'config')
user_password_info_dict_filename = conf_yaml['user']['user_password_info_dict_path']
user_password_info_dict_path = os.path.join(config_dir, user_password_info_dict_filename)

'''parse voice config'''
voice_chunk_size = conf_yaml['rag']['parse_voice']['voice_chunk_size']


@singleton
class Deal_user_article_kb_config():
    def __init__(self, database_type):
        '''数据库相关'''
        self.database_root_dir = self.init_data_dir()
        self.get_mysql_config()
        self.get_lancedb_config()
        self.get_milvus_config()
        self.get_es_config()
        self.get_database_config(database_type)
        self.init_article_user_and_kb_mapping_file()

    def init_data_dir(self):
        project_dir = os.path.dirname(os.path.dirname(__file__))
        data_dir = os.path.join(project_dir, 'data')
        database_root_dir = os.path.join(data_dir, 'database')
        return database_root_dir

    def get_mysql_config(self):
        '''mysql配置'''
        self.mysql_host = conf_yaml['mysql']['host']
        self.mysql_port = conf_yaml['mysql']['port']
        self.mysql_user = conf_yaml['mysql']['user']
        self.mysql_password = conf_yaml['mysql']['password']
        self.mysql_database_name = conf_yaml['mysql']['database_name']
        mysql_data_dir = os.path.join(self.database_root_dir, 'mysql')
        self.mysql_article_table_info_name = conf_yaml['mysql']['article_table_name']
        self.mysql_articles_user_path = os.path.join(mysql_data_dir, 'user_article_mapping.json')
        self.mysql_kb_article_map_path = os.path.join(mysql_data_dir, 'kb_article_mappping.json')


    def get_lancedb_config(self):
        '''lancedb配置相关'''
        self.lancedb_data_dir = os.path.join(self.database_root_dir, 'lancedb')
        self.lancedb_articles_user_path = os.path.join(self.lancedb_data_dir, 'user_article_mapping.json')
        self.lancedb_kb_article_map_path = os.path.join(self.lancedb_data_dir, 'kb_article_mappping.json')

    def get_milvus_config(self):
        '''milvus配置相关'''
        milvus_data_dir = os.path.join(self.database_root_dir, 'milvus')
        self.milvus_articles_user_path = os.path.join(milvus_data_dir, 'user_article_mapping.json')
        self.milvus_kb_article_map_path = os.path.join(milvus_data_dir, 'kb_article_mappping.json')

    def get_es_config(self):
        es_data_dir = os.path.join(self.database_root_dir, 'es')
        self.es_articles_user_path = os.path.join(es_data_dir, 'user_article_mapping.json')
        self.es_kb_article_map_path = os.path.join(es_data_dir, 'kb_article_mappping.json')
        self.es_index_name = 'article_index'

    def get_username(self, request: gr.Request):
        '''获取初始状态'''
        article_dict = read_user_info_dict(request.username, self.articles_user_path)  # id:value
        kb_article_dict = read_user_info_dict(request.username, self.kb_article_map_path)
        return article_dict, kb_article_dict

    def database_type_dropdowns(self, input_value, request: gr.Request):
        choices = ['lancedb', 'milvus', 'mysql', 'es']
        self.get_database_config(database_type=input_value)
        dropdown = gr.Dropdown(choices=choices, label='关联的数据库', value=input_value, interactive=True)
        article_dict, kb_article_dict = self.get_username(request)
        return dropdown, dropdown, dropdown, article_dict, kb_article_dict

    def get_database_config(self, database_type='lancedb'):
        '''根据数据库进行初始化配置'''
        if database_type == 'lancedb':
            database_dir = os.path.join(self.lancedb_data_dir, 'data')
            articles_user_path = self.lancedb_articles_user_path
            kb_article_map_path = self.lancedb_kb_article_map_path
        elif database_type == 'milvus':
            database_dir = ''
            articles_user_path = self.milvus_articles_user_path
            kb_article_map_path = self.milvus_kb_article_map_path
        elif database_type == 'mysql':
            database_dir = ''
            articles_user_path = self.mysql_articles_user_path
            kb_article_map_path = self.mysql_kb_article_map_path
        elif database_type == 'es':
            database_dir = ''
            articles_user_path = self.es_articles_user_path
            kb_article_map_path = self.es_kb_article_map_path
        else:
            assert False, f"{database_type} not support!"
        self.database_dir = database_dir
        self.articles_user_path = articles_user_path
        self.kb_article_map_path = kb_article_map_path
        self.database_type = database_type

    def init_article_user_and_kb_mapping_file(self):
        '''创建user_article_mapping.json和kb_article_mappping.json文件'''
        user_info_dict = read_json_file(user_password_info_dict_path)
        if os.path.exists(self.articles_user_path):
            pass
        else:
            os.makedirs(os.path.dirname(self.articles_user_path), exist_ok=True)
            articles_user_default_json = {}
            for user_name in user_info_dict.keys():
                articles_user_default_json[user_name] = {}
            save_json_file(articles_user_default_json, self.articles_user_path)

        if os.path.exists(self.kb_article_map_path):
            pass
        else:
            os.makedirs(os.path.dirname(self.kb_article_map_path), exist_ok=True)
            kb_article_map_default_json = {}
            for user_name in user_info_dict.keys():
                kb_article_map_default_json[user_name] = {}
            save_json_file(kb_article_map_default_json, self.kb_article_map_path)

akb_conf_class = Deal_user_article_kb_config('lancedb')

articles_user_path = akb_conf_class.articles_user_path
kb_article_map_path = akb_conf_class.kb_article_map_path
database_dir = akb_conf_class.database_dir
database_type =akb_conf_class.database_type
