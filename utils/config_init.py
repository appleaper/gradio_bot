import os
import torch
import gradio as gr
from utils.tool import read_user_info_dict, singleton
from utils.tool import save_json_file, read_json_file

@singleton
class Config():
    def __init__(self):
        self.project_dir = os.path.dirname(os.path.dirname(__file__))

    def init_config(self, username):
        '''初始化全部配置'''
        self.config_dict = self.read_local_default_config()[username]
        self.init_article_user_and_kb_mapping_file()
        self.gpu_is_available()
        self.init_mid_data_dir()
        self.config_dict['user_password_path'] = self.get_password_path()
        self.config_dict['voice_chunk_size'] = 1024
        self.config_dict['project_dir'] = self.project_dir

    def read_local_default_config(self):
        '''加载默认配置'''
        local_config_path = os.path.join(self.project_dir, 'config', 'config.yaml')
        config_info_dict = read_json_file(local_config_path)
        return config_info_dict

    def gpu_is_available(self):
        '''cuda设备选择'''
        if torch.cuda.is_available():
            device = torch.device('cuda')
            device_str = 'cuda'
        else:
            device = torch.device('cpu')
            device_str = 'cpu'
        self.config_dict['device'] = device
        self.config_dict['device_str'] = device_str

    def init_mid_data_dir(self):
        '''中间数据'''
        data_dir = os.path.join(self.project_dir, 'data')
        tmp_dir = os.path.join(data_dir, 'tmp')
        font_path = os.path.join(data_dir, 'ocr', 'font', 'SimHei.ttf')
        self.config_dict['tmp_dir'] = tmp_dir
        self.config_dict['font_path'] = font_path

    def get_password_path(self):
        '''用户账号密码'''
        config_dir = os.path.join(self.project_dir, 'config')
        user_password_info_dict_path = os.path.join(config_dir, 'auth.json')
        return user_password_info_dict_path

    def get_user_id2article_and_article_group(self, db_name, user_name, embedding_type):
        '''获得id2文章和知识库组字典'''
        id2article_path = self.config_dict[db_name]['id2article']
        article_group_path = self.config_dict[db_name]['article_group']
        id2article_dict_info = read_json_file(id2article_path)
        article_group_dict_info = read_json_file(article_group_path)
        if user_name in id2article_dict_info:
            if embedding_type in id2article_dict_info[user_name]:
                id2article_info = id2article_dict_info[user_name][embedding_type]
            else:
                id2article_info = {}
        else:
            id2article_info = {}

        if user_name in article_group_dict_info:
            if embedding_type in article_group_dict_info[user_name]:
                article_group_info = article_group_dict_info[user_name][embedding_type]
            else:
                article_group_info = {}
        else:
            article_group_info = {}

        return id2article_info, article_group_info

    def init_article_user_and_kb_mapping_file(self):
        '''创建user_article_mapping.json和kb_article_mappping.json文件'''
        user_password_info_dict_path = self.get_password_path()
        user_info_dict = read_json_file(user_password_info_dict_path)
        for db in self.config_dict['database_type']:
            for file_name in ['id2article', 'article_group']:
                create_path = self.config_dict[db][file_name]
                if os.path.exists(create_path):
                    pass
                else:
                    info = {}
                    for user_name in user_info_dict.keys():
                        info[user_name] = {}
                    save_json_file(info, create_path)

conf_class = Config()
