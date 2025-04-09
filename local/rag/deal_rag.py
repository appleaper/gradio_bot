import copy
import os
import time
import lancedb
import hashlib
import gradio as gr
import pandas as pd
from gradio import ChatMessage
from utils.tool import singleton, read_json_file, save_json_file, reverse_dict, encrypt_username

from local.model.emb_model.bge_m3 import EmbBgeM3
from local.model.chat_model.qwen_model import QwenChatModel
from local.model.ollama_model.ollama_func import OllamaClient

from local.database.mysql.mysql_article_management import MySQLDatabase
from local.database.milvus.milvus_article_management import MilvusArticleManager
from local.database.es.es_article_management import ElasticsearchManager
from local.database.lancedb.data_to_lancedb import create_or_add_data_to_lancedb

from local.rag.parse.pdf_parse import parse_pdf_do
from local.rag.parse.csv_parse import parse_csv_do
from local.rag.parse.docx_parser import parse_docx_do
from local.rag.parse.markdown_parse import parse_markdown_do
from local.rag.parse.image_parse import parse_image_do
from local.rag.parse.voice_parse import parse_voice_do
from local.rag.parse.video_parse import parse_video_do

@singleton
class Deal_emb():
    def __init__(self):
        pass

    def load_model(self, rag_embedding_type, config_info):
        if rag_embedding_type.startswith('local'):
            if rag_embedding_type.endswith('bge-m3'):
                emb_model_class = EmbBgeM3(config_info['device_str'])
                emb_model_class.load_model(config_info['local_model_name_path_dict'][rag_embedding_type])
            else:
                raise gr.Error('其他本地embedding模型暂时还不支持')
        elif rag_embedding_type.startswith('ollama'):
            emb_model_class = OllamaClient(config_info['ollama']['host'], config_info['ollama']['port'])
            emb_model_class.model_name = rag_embedding_type[7:]
        else:
            raise gr.Error(f'{rag_embedding_type} 暂时还不支持')
        return emb_model_class

@singleton
class ParseFileType():
    def __init__(self):
        self.emb_class = Deal_emb()

    def parse_csv(self, file_path, user_id, database_type, embedding_class):
        df = parse_csv_do(file_path, user_id, database_type, embedding_class)
        return df

    def parse_docx(self, file_path, user_id, database_type, emb_model_class):
        df = parse_docx_do(file_path, user_id, database_type, emb_model_class)
        return df

    def parse_image(self, file_path, user_id, database_type, emb_model_class, config_info):
        df = parse_image_do(file_path, user_id, database_type, emb_model_class, config_info)
        return df

    def parse_markdown(self, file_path, user_id, database_type, emb_model_class):
        df = parse_markdown_do(file_path, user_id, database_type, emb_model_class)
        return df

    def parse_pdf(self, file_path, user_id, database_type, embedding_class, config_info):
        df = parse_pdf_do(file_path, user_id, database_type, embedding_class, config_info)
        return df

    def parse_video(self, file_path, user_id, database_type, embedding_class, config_info):
        df = parse_video_do(file_path, user_id, database_type, embedding_class, config_info)
        return df

    def parse_voice(self, file_path, user_id, database_type, embedding_class, config_info):
        df = parse_voice_do(file_path, user_id, database_type, embedding_class, config_info)
        return df

    def parse_xlsx(self, file_path, user_id, database_type, embedding_class):
        df = parse_csv_do(file_path, user_id, database_type, embedding_class)
        return df

    def parse_single_file(self):
        pass

    def parse_many_files(self, files_path, database_type, embedding_type, config_info):
        self.emb_model_class = self.emb_class.load_model(embedding_type, config_info)
        user_id = config_info['user_id']
        result_list = []
        for file_path in files_path:
            filname, suffix = os.path.splitext(os.path.basename(file_path))
            if suffix == '.pdf':
                df = self.parse_pdf(file_path, user_id, database_type, self.emb_model_class, config_info)
            elif suffix == '.csv':
                df = self.parse_csv(file_path, user_id, database_type, self.emb_model_class)
            elif suffix == '.xlsx':
                df = self.parse_xlsx(file_path, user_id, database_type, self.emb_model_class)
            elif suffix == '.md':
                df = self.parse_markdown(file_path, user_id, database_type, self.emb_model_class)
            elif suffix == '.docx':
                df = self.parse_docx(file_path, user_id, database_type, self.emb_model_class)
            elif suffix in ['.jpg', '.jpeg', '.png']:
                df = self.parse_image(file_path, user_id, database_type, self.emb_model_class, config_info)
            elif suffix == '.mp3':
                df = self.parse_voice(file_path, user_id, database_type, self.emb_model_class, config_info)
            elif suffix == '.mp4':
                df = self.parse_video(file_path, user_id, database_type, self.emb_model_class, config_info)
            else:
                gr.Warning(f'{os.path.basename(file_path)}不支持解析')
                continue
            result_list.append(df)
        return result_list

@singleton
class DealDataToDB():
    def __init__(self):
        self.emb_class = Deal_emb()

    def init_mysql(self, mysql_host, mysql_user, mysql_password, mysql_port, mysql_database_name):
        '''插入数据的时候，初始化mysql'''
        manager = MySQLDatabase(
            host=mysql_host,
            user=mysql_user,
            password=mysql_password,
            port=mysql_port
        )
        manager.connect()
        manager.create_database(mysql_database_name)
        manager.use_database(mysql_database_name)
        return manager

    def create_article_id(self, is_same_group, article_name, df_list):
        article_name = article_name.replace(' ', '')
        if article_name=='':
            temp_name = os.path.splitext(os.path.basename(df_list[0].iloc[0]['file_from']))[0]
        else:
            temp_name = article_name
        temp_id = hashlib.sha256((temp_name).encode('utf-8')).hexdigest()
        id_list = []
        name_list = []
        for df in df_list:
            if is_same_group:
                id_list.append(temp_id)
                name_list.append(temp_name)
            else:
                article_name = os.path.splitext(os.path.basename(df.iloc[0]['file_from']))[0]
                name_list.append(article_name)
                id_list.append(hashlib.sha256((article_name).encode('utf-8')).hexdigest())
        return id_list, name_list

    def update_id2article_dict(self, id2article_dict, config_info, new_article_dict, rag_embedding_type):
        username = config_info['username']
        for art_id, art_name in new_article_dict.items():
            if username not in id2article_dict:
                id2article_dict[username] = {}
                if rag_embedding_type not in id2article_dict[username]:
                    id2article_dict[username][rag_embedding_type] = {}
                    id2article_dict[username][rag_embedding_type][art_id] = art_name
                else:
                    id2article_dict[username][rag_embedding_type][art_id] = art_name
            else:
                if rag_embedding_type not in id2article_dict[username]:
                    id2article_dict[username][rag_embedding_type] = {}
                    id2article_dict[username][rag_embedding_type][art_id] = art_name
                else:
                    id2article_dict[username][rag_embedding_type][art_id] = art_name
        return id2article_dict

    def save_df_to_mysql(self, rag_embedding_type, info_list, is_same_group, article_name, config_info):
        id_list, name_list = self.create_article_id(is_same_group, article_name, info_list)
        save_all_df = pd.DataFrame()
        new_article_dict = {}
        manager = self.init_mysql(
            mysql_host=config_info['mysql']['host'],
            mysql_user=config_info['mysql']['user'],
            mysql_password=config_info['mysql']['password'],
            mysql_port=config_info['mysql']['port'],
            mysql_database_name=config_info['mysql']['database'],
        )
        mysql_table_name = config_info['mysql']['table']
        for index, df in enumerate(info_list):
            article_id = id_list[index]
            name = name_list[index]
            new_article_dict[article_id] = name
            save_i_df = manager.insert_data_format_df(mysql_table_name, df)
            save_all_df = pd.concat((save_i_df, save_all_df))
        id2article_dict = read_json_file(config_info['mysql']['id2article'])
        id2article_dict = self.update_id2article_dict(id2article_dict, config_info, new_article_dict, rag_embedding_type)
        username = config_info['username']
        save_json_file(id2article_dict, config_info['mysql']['id2article'])
        id2article_dict = read_json_file(config_info['mysql']['id2article'])[username][rag_embedding_type]
        config_info['id2article_dict'] = id2article_dict
        return config_info

    def save_df_to_milvus(self, rag_embedding_type, df_list, is_same_group, article_name, config_info):
        id_list, name_list = self.create_article_id(is_same_group, article_name, df_list)
        manager = MilvusArticleManager(host=config_info['milvus']['host'], port=config_info['milvus']['port'])
        user_id = encrypt_username(config_info['username'])
        manager.create_collection(user_id)
        save_all_df = pd.DataFrame()
        new_article_dict = {}
        for index, df in enumerate(df_list):
            article_id = id_list[index]
            name = name_list[index]
            new_article_dict[article_id] = name
            save_i_df = manager.insert_data_to_milvus(df, user_id)
            save_all_df = pd.concat((save_i_df, save_all_df))
        id2article_dict = read_json_file(config_info['milvus']['id2article'])
        id2article_dict = self.update_id2article_dict(id2article_dict, config_info, new_article_dict, rag_embedding_type)
        username = config_info['username']
        save_json_file(id2article_dict, config_info['milvus']['id2article'])
        id2article_dict = read_json_file(config_info['milvus']['id2article'])[username][rag_embedding_type]
        config_info['id2article_dict'] = id2article_dict
        return config_info

    def save_df_to_lancedb(self, rag_embedding_type, df_list, is_same_group, article_name, config_info):
        '''数据保存到lancedb中'''
        id_list, name_list = self.create_article_id(is_same_group, article_name, df_list)
        database_name = config_info['lancedb']['data_dir']
        save_all_df = pd.DataFrame()
        new_article_dict = {}
        for index, df in enumerate(df_list):
            article_id = id_list[index]
            name = name_list[index]
            new_article_dict[article_id] = name
            save_i_df = create_or_add_data_to_lancedb(database_name, article_id, df)
            save_all_df = pd.concat((save_i_df, save_all_df))
        id2article_dict = read_json_file(config_info['lancedb']['id2article'])
        id2article_dict = self.update_id2article_dict(id2article_dict, config_info, new_article_dict, rag_embedding_type)
        username = config_info['username']
        save_json_file(id2article_dict, config_info['lancedb']['id2article'])
        id2article_dict = read_json_file(config_info['lancedb']['id2article'])[username][rag_embedding_type]
        config_info['id2article_dict'] = id2article_dict
        return config_info

    def save_df_to_es(self, rag_embedding_type, info_list, is_same_group, article_name, config_info):
        manager = ElasticsearchManager(
            index_name=config_info['es']['index_name'],
            host=config_info['es']['host'],
            port=config_info['es']['port'],
            scheme=config_info['es']['scheme']
        )
        id_list, name_list = self.create_article_id(is_same_group, article_name, info_list)
        save_all_df = pd.DataFrame()
        new_article_dict = {}
        for index, df in enumerate(info_list):
            article_id = id_list[index]
            name = name_list[index]
            new_article_dict[article_id] = name
            save_i_df = manager.insert_data_format_df(df)
            save_all_df = pd.concat((save_i_df, save_all_df))
        id2article_dict = read_json_file(config_info['es']['id2article'])
        id2article_dict = self.update_id2article_dict(id2article_dict, config_info, new_article_dict, rag_embedding_type)
        username = config_info['username']
        save_json_file(id2article_dict, config_info['es']['id2article'])
        id2article_dict = read_json_file(config_info['es']['id2article'])[username][rag_embedding_type]
        config_info['id2article_dict'] = id2article_dict
        return config_info

    def save_df_to_db(self, rag_database_type, rag_embedding_type, info_list, is_same_group, article_name, config_info):
        if rag_database_type == 'lancedb':
            config_info = self.save_df_to_lancedb(rag_embedding_type, info_list, is_same_group, article_name, config_info)
        elif rag_database_type == 'milvus':
            config_info = self.save_df_to_milvus(rag_embedding_type, info_list, is_same_group, article_name, config_info)
        elif rag_database_type == 'mysql':
            config_info = self.save_df_to_mysql(rag_embedding_type, info_list, is_same_group, article_name, config_info)
        elif rag_database_type == 'es':
            config_info = self.save_df_to_es(rag_embedding_type, info_list, is_same_group, article_name, config_info)
        return config_info

    def delete_df_to_lancedb(self, rag_database_type, rag_embedding_type, rag_checkboxgroup, config_info):
        '''从lancedb中删除数据'''
        data_dir = config_info[rag_database_type]['data_dir']
        id2article_path = config_info[rag_database_type]['id2article']
        db = lancedb.connect(data_dir)
        id2article_info = read_json_file(id2article_path)
        username = config_info['username']
        id2article_dict, article2id_dict = self.flip_id_article_dict(username, id2article_info, rag_embedding_type)

        for article in rag_checkboxgroup:
            if article not in article2id_dict:
                continue
            else:
                id = str(article2id_dict[article])
                table_path = os.path.join(data_dir, id + '.lance')
                if os.path.exists(table_path):
                    db.drop_table(id)
                    del id2article_dict[id]
                    gr.Info(f'成功删除{article}')
                else:
                    del id2article_dict[id]
                    gr.Warning(f'数据不存在，但依旧执行删除')
        id2article_info[username][rag_embedding_type] = id2article_dict
        save_json_file(id2article_info, id2article_path)
        config_info['id2article_dict'] = id2article_dict
        article_group_path = config_info[rag_database_type]['article_group']
        group_articles_dict_copy = self.delete_article_from_group(article_group_path, username, rag_embedding_type, rag_checkboxgroup)
        config_info['article_group_dict'] = group_articles_dict_copy
        return config_info

    def flip_id_article_dict(self, username, id2article_info, rag_embedding_type):
        if username not in id2article_info:
            id2article_info[username] = {}
        if rag_embedding_type not in id2article_info[username]:
            id2article_info[username][rag_embedding_type] = {}
        id2article_dict = id2article_info[username][rag_embedding_type]
        article2id_dict = reverse_dict(id2article_dict)
        return id2article_dict, article2id_dict

    def delete_article_from_group(self, article_group_path, username, rag_embedding_type, rag_checkboxgroup):
        article_group_json = read_json_file(article_group_path)
        if username not in article_group_json:
            article_group_json[username] = {}
        if rag_embedding_type not in article_group_json[username]:
            article_group_json[username][rag_embedding_type] = {}
        group_articles_dict = article_group_json[username][rag_embedding_type]
        group_articles_dict_copy = copy.deepcopy(group_articles_dict)
        for group_name, article_list in group_articles_dict.items():
            left_list = []
            for article in article_list:
                if article in rag_checkboxgroup:
                    continue
                else:
                    left_list.append(article)
            if len(left_list) == 0:
                del group_articles_dict_copy[group_name]
            else:
                group_articles_dict_copy[group_name] = left_list
        article_group_json[username][rag_embedding_type] = group_articles_dict_copy
        save_json_file(article_group_json, article_group_path)
        return group_articles_dict_copy

    def delete_df_to_milvus(self, rag_database_type, rag_embedding_type, rag_checkboxgroup, config_info):
        id2article_path = config_info[rag_database_type]['id2article']
        id2article_info = read_json_file(id2article_path)
        username = config_info['username']
        id2article_dict, article2id_dict = self.flip_id_article_dict(username, id2article_info, rag_embedding_type)

        manager = MilvusArticleManager(host=config_info['milvus']['host'], port=config_info['milvus']['port'])
        user_id = encrypt_username(config_info['username'])
        need_delete_articles_id_list = []
        for article in rag_checkboxgroup:
            if article not in article2id_dict:
                continue
            else:
                id = str(article2id_dict[article])
                need_delete_articles_id_list.append(id)
                if id in id2article_dict:
                    del id2article_dict[id]
        manager.delete_data_by_article_id(user_id, need_delete_articles_id_list)

        id2article_info[username][rag_embedding_type] = id2article_dict
        save_json_file(id2article_info, id2article_path)
        config_info['id2article_dict'] = id2article_dict
        article_group_path = config_info[rag_database_type]['article_group']

        group_articles_dict_copy = self.delete_article_from_group(article_group_path, username, rag_embedding_type,
                                                                  rag_checkboxgroup)
        config_info['article_group_dict'] = group_articles_dict_copy
        return config_info

    def delete_df_to_mysql(self, rag_database_type, rag_embedding_type, rag_checkboxgroup, config_info):
        manager = self.init_mysql(
            mysql_host=config_info['mysql']['host'],
            mysql_user=config_info['mysql']['user'],
            mysql_password=config_info['mysql']['password'],
            mysql_port=config_info['mysql']['port'],
            mysql_database_name=config_info['mysql']['database'],
        )
        mysql_table_name = config_info['mysql']['table']
        user_id = encrypt_username(config_info['username'])

        id2article_path = config_info[rag_database_type]['id2article']
        id2article_info = read_json_file(id2article_path)
        username = config_info['username']
        id2article_dict, article2id_dict = self.flip_id_article_dict(username, id2article_info, rag_embedding_type)

        need_delete_articles_id_list = []
        for article in rag_checkboxgroup:
            if article not in article2id_dict:
                continue
            else:
                id = str(article2id_dict[article])
                need_delete_articles_id_list.append(id)
                if id in id2article_dict:
                    del id2article_dict[id]
        manager.delete_data_by_user_and_article_ids(mysql_table_name, user_id, need_delete_articles_id_list)

        id2article_info[username][rag_embedding_type] = id2article_dict
        save_json_file(id2article_info, id2article_path)
        config_info['id2article_dict'] = id2article_dict
        article_group_path = config_info[rag_database_type]['article_group']

        group_articles_dict_copy = self.delete_article_from_group(article_group_path, username, rag_embedding_type,
                                                                  rag_checkboxgroup)
        config_info['article_group_dict'] = group_articles_dict_copy
        return config_info

    def delete_df_to_es(self, rag_database_type, rag_embedding_type, rag_checkboxgroup, config_info):
        manager = ElasticsearchManager(
            index_name=config_info['es']['index_name'],
            host=config_info['es']['host'],
            port=config_info['es']['port'],
            scheme=config_info['es']['scheme']
        )
        user_id = encrypt_username(config_info['username'])

        id2article_path = config_info[rag_database_type]['id2article']
        id2article_info = read_json_file(id2article_path)
        username = config_info['username']
        id2article_dict, article2id_dict = self.flip_id_article_dict(username, id2article_info, rag_embedding_type)

        need_delete_articles_id_list = []
        for article in rag_checkboxgroup:
            if article not in article2id_dict:
                continue
            else:
                id = str(article2id_dict[article])
                need_delete_articles_id_list.append(id)
                if id in id2article_dict:
                    del id2article_dict[id]
        manager.delete_data(user_id, need_delete_articles_id_list)

        id2article_info[username][rag_embedding_type] = id2article_dict
        save_json_file(id2article_info, id2article_path)
        config_info['id2article_dict'] = id2article_dict
        article_group_path = config_info[rag_database_type]['article_group']

        group_articles_dict_copy = self.delete_article_from_group(article_group_path, username, rag_embedding_type,
                                                                  rag_checkboxgroup)
        config_info['article_group_dict'] = group_articles_dict_copy
        return config_info

    def delete_df_to_db(self, rag_database_type, rag_embedding_type, rag_checkboxgroup, config_info):
        if rag_database_type == 'lancedb':
            config_info = self.delete_df_to_lancedb(rag_database_type, rag_embedding_type, rag_checkboxgroup, config_info)
        elif rag_database_type == 'milvus':
            config_info = self.delete_df_to_milvus(rag_database_type, rag_embedding_type, rag_checkboxgroup, config_info)
        elif rag_database_type == 'mysql':
            config_info = self.delete_df_to_mysql(rag_database_type, rag_embedding_type, rag_checkboxgroup, config_info)
        elif rag_database_type == 'es':
            config_info = self.delete_df_to_es(rag_database_type, rag_embedding_type, rag_checkboxgroup, config_info)
        return config_info

    def get_input_text_embedding(self, search_text):
        emb_model_name = self.embedding_class.model_name
        vector = self.embedding_class.parse_single_sentence(
            model_name=emb_model_name,
            sentence=search_text
        )[0]
        return vector

    def get_article_group(self, config_info, database_type, embedding_type, username):
        article_group_path = config_info[database_type]['article_group']
        article_group_dict = read_json_file(article_group_path)
        if username not in article_group_dict:
            article_group_dict[username] = {}
        if embedding_type not in article_group_dict[username]:
            article_group_dict[username][database_type] = {}
        return article_group_dict

    def select_df_to_lancedb(self, search_database_type, search_embedding_type, search_kb_range, search_tok_k, search_text, config_info):
        vector = self.get_input_text_embedding(search_text)
        data_dir = config_info[search_database_type]['data_dir']

        id2article_path = config_info[search_database_type]['id2article']
        id2article_info = read_json_file(id2article_path)
        username = config_info['username']
        id2article_dict, article2id_dict = self.flip_id_article_dict(username, id2article_info, search_embedding_type)
        article_group_all_dict = self.get_article_group(config_info, search_database_type, search_embedding_type, username)
        article_group_dict = article_group_all_dict[username][search_embedding_type]
        db = lancedb.connect(data_dir)
        records_df = pd.DataFrame()
        search_tok_k = int(search_tok_k)
        for article_name in article_group_dict[search_kb_range]:
            tb = db.open_table(article2id_dict[article_name])
            records = tb.search(vector).distance_type("cosine").limit(search_tok_k).to_pandas()
            records_df = pd.concat((records, records_df))
        if len(records_df) == 0:
            return pd.DataFrame([])
        else:
            sorted_records_df = records_df.sort_values(by='_distance').iloc[:search_tok_k]
            result_list = []
            for _, row in sorted_records_df.iterrows():
                info = {}
                info['title'] = row['title']
                info['content'] = row['content']
                info['file_from'] = row['file_from']
                info['score'] = row['_distance']
                result_list.append(info)
            return pd.DataFrame(result_list)

    def select_df_to_milvus(self, search_database_type, search_embedding_type, search_kb_range, search_tok_k, search_text, config_info):
        vector = self.get_input_text_embedding(search_text)
        id2article_path = config_info[search_database_type]['id2article']
        id2article_info = read_json_file(id2article_path)
        username = config_info['username']
        id2article_dict, article2id_dict = self.flip_id_article_dict(username, id2article_info, search_embedding_type)
        article_group_all_dict = self.get_article_group(config_info, search_database_type, search_embedding_type, username)
        article_group_dict = article_group_all_dict[username][search_embedding_type]

        manager = MilvusArticleManager(host=config_info['milvus']['host'], port=config_info['milvus']['port'])
        user_id = encrypt_username(config_info['username'])
        articles_id_list = []
        for article_name in article_group_dict[search_kb_range]:
            articles_id_list.append(article2id_dict[article_name])
        emb_model_name = self.embedding_class.model_name
        res = manager.search_vectors_with_articles(user_id, emb_model_name, vector, articles_id_list, limit=search_tok_k)
        df = pd.DataFrame(res)
        return df

    def select_df_to_mysql(self, search_database_type, search_embedding_type, search_kb_range, search_tok_k, search_text, config_info):
        id2article_path = config_info[search_database_type]['id2article']
        id2article_info = read_json_file(id2article_path)
        username = config_info['username']
        id2article_dict, article2id_dict = self.flip_id_article_dict(username, id2article_info, search_embedding_type)
        article_group_all_dict = self.get_article_group(config_info, search_database_type, search_embedding_type,
                                                        username)
        article_group_dict = article_group_all_dict[username][search_embedding_type]

        manager = self.init_mysql(
            mysql_host=config_info['mysql']['host'],
            mysql_user=config_info['mysql']['user'],
            mysql_password=config_info['mysql']['password'],
            mysql_port=config_info['mysql']['port'],
            mysql_database_name=config_info['mysql']['database'],
        )
        table_name = config_info['mysql']['table']


        user_id = encrypt_username(config_info['username'])
        articles_id_list = []
        for article_name in article_group_dict[search_kb_range]:
            articles_id_list.append(article2id_dict[article_name])

        df = manager.select_data(table_name, user_id, search_text, articles_id_list, search_tok_k)
        return df

    def select_df_to_es(self, search_database_type, search_embedding_type, search_kb_range, search_tok_k, search_text,
                                           config_info):
        id2article_path = config_info[search_database_type]['id2article']
        id2article_info = read_json_file(id2article_path)
        username = config_info['username']
        id2article_dict, article2id_dict = self.flip_id_article_dict(username, id2article_info, search_embedding_type)
        article_group_all_dict = self.get_article_group(config_info, search_database_type, search_embedding_type,
                                                        username)
        article_group_dict = article_group_all_dict[username][search_embedding_type]
        manager = ElasticsearchManager(
            index_name=config_info['es']['index_name'],
            host=config_info['es']['host'],
            port=config_info['es']['port'],
            scheme=config_info['es']['scheme']
        )
        user_id = encrypt_username(config_info['username'])
        articles_id_list = []
        for article_name in article_group_dict[search_kb_range]:
            articles_id_list.append(article2id_dict[article_name])

        result = manager.query_data(user_id, articles_id_list, search_text, search_tok_k)
        df = pd.DataFrame(result)
        if len(df)!=0:
            df = df[['title', 'content', 'file_from']]
            df['score'] = 1
        else:
            df = pd.DataFrame([])
        return df

    def select_df_to_db(self, search_database_type, search_embedding_type, search_kb_range, search_tok_k, search_text, config_info):
        self.embedding_class = self.emb_class.load_model(search_embedding_type, config_info)
        if search_database_type=='lancedb':
            df = self.select_df_to_lancedb(search_database_type, search_embedding_type, search_kb_range, search_tok_k, search_text, config_info)
        elif search_database_type == 'milvus':
            df = self.select_df_to_milvus(search_database_type, search_embedding_type, search_kb_range, search_tok_k, search_text,
                                           config_info)
        elif search_database_type == 'mysql':
            df = self.select_df_to_mysql(search_database_type, search_embedding_type, search_kb_range, search_tok_k, search_text,
                                           config_info)
        elif search_database_type == 'es':
            df = self.select_df_to_es(search_database_type, search_embedding_type, search_kb_range, search_tok_k, search_text,
                                           config_info)
        else:
            df = pd.DataFrame([])
            gr.Warning(f'不支持这个数据库')
        if len(df) != 0:
            df = df.drop_duplicates(subset=['content'])
        return df

@singleton
class DealRag():
    def __init__(self):
        self.parse_class = ParseFileType()
        self.db_class = DealDataToDB()

    def delete_article(self, rag_database_type, rag_embedding_type, rag_checkboxgroup, config_info):
        '''
        删除文章
        :param rag_database_type: 数据库
        :param rag_embedding_type: 编码方式
        :param rag_checkboxgroup: 要删除的文章
        :param config_info: 全局配置
        :return:
        '''
        config_info = self.db_class.delete_df_to_db(rag_database_type, rag_embedding_type, rag_checkboxgroup, config_info)
        if len(rag_checkboxgroup) == 0:
            book_type = gr.Dropdown(choices=[], label="上下文知识")
            selectable_documents_checkbox_group = gr.CheckboxGroup(choices=[], label='可选文章', interactive=True)
            selectable_knowledge_bases_checkbox_group = gr.CheckboxGroup(choices=[], label='已有知识库', interactive=True)
            search_kb_range = gr.Dropdown(choices=[], label='检索范围', interactive=True)
            knowledge_base_info_json_table = gr.JSON(value={})
        else:
            article_list = list(config_info['id2article_dict'].values())
            group_list = list(config_info['article_group_dict'].keys())
            book_type = gr.Dropdown(choices=group_list, label="上下文知识")
            selectable_documents_checkbox_group = gr.CheckboxGroup(choices=article_list, label='可选文章', interactive=True)
            selectable_knowledge_bases_checkbox_group = gr.CheckboxGroup(choices=group_list, label='已有知识库',
                                                                         interactive=True)
            search_kb_range = gr.Dropdown(choices=group_list, label='检索范围', interactive=True)
            knowledge_base_info_json_table = gr.JSON(value=config_info['article_group_dict'])
        return [], book_type, selectable_documents_checkbox_group, selectable_knowledge_bases_checkbox_group, search_kb_range, knowledge_base_info_json_table, config_info

    def add_article(self, rag_database_type, rag_embedding_type, is_same_group, article_name, rag_upload_file, config_info):
        '''
        添加文章
        :param rag_database_type: 数据库
        :param rag_embedding_type: 编码方式
        :param is_same_group: 是否为同一组
        :param article_name: 组名
        :param rag_upload_file: 上传的文件
        :param config_info: 全局配置
        :return:
        rag_checkboxgroup, selectable_documents_checkbox_group, config_info
        '''
        is_same_group = is_same_group == '是'
        parser_info_list = self.parse_class.parse_many_files(rag_upload_file, rag_database_type, rag_embedding_type, config_info)
        config_info = self.db_class.save_df_to_db(rag_database_type, rag_embedding_type, parser_info_list, is_same_group, article_name, config_info)
        article_list = list(config_info['id2article_dict'].values())
        rag_checkboxgroup = gr.CheckboxGroup(choices=article_list, label="rag列表", interactive=True)
        selectable_documents_checkbox_group = gr.CheckboxGroup(choices=article_list, label='可选文章', interactive=True)
        config_json = gr.JSON(value=config_info,visible=False)
        return rag_checkboxgroup, selectable_documents_checkbox_group, '', '', [], config_json

    def update_group_info(self, info, user, embed_type, kb_name, docs):
        if user not in info:
            info[user] = {}
        if embed_type not in info[user]:
            info[user][embed_type] = {}
        if kb_name=='':
            return info
        elif len(docs) == 0:
            return info
        else:
            info[user][embed_type][kb_name] = docs
            return info

    def delete_group_entries(self, info, user, emb_type, grp_names):
        if user not in info:
            info[user] = {}
        if emb_type not in info[user]:
            info[user][emb_type] = {}
        for name in grp_names:
            if name in info[user][emb_type]:
                del info[user][emb_type][name]
        return info

    def add_article_group(self, kb_database_type, kb_embedding_type, selectable_documents_checkbox_group, new_kb_name_textbox, config_info):
        '''
        新建知识库
        :param kb_database_type: 数据库名字
        :param kb_embedding_type: 编码类型
        :param selectable_documents_checkbox_group: 可选文章
        :param new_kb_name_textbox: 知识库名
        :param config_info: 配置信息
        :return:
        '''
        article_group_path = config_info[kb_database_type]['article_group']
        group_info_json = read_json_file(article_group_path)
        username = config_info['username']
        if new_kb_name_textbox == '':
            gr.Warning('请填入知识库名字')
        if len(selectable_documents_checkbox_group) == 0:
            gr.Warning('请选择知识库的文章列表')

        group_info_json = self.update_group_info(group_info_json, username, kb_embedding_type, new_kb_name_textbox, selectable_documents_checkbox_group)
        save_json_file(group_info_json, article_group_path)
        article_list = group_info_json[username][kb_embedding_type].keys()
        config_info['article_group_dict'] = group_info_json

        selectable_knowledge_bases_checkbox_group = gr.CheckboxGroup(choices=article_list, label='已有知识库', interactive=True)
        book_type = gr.Dropdown(choices=article_list, label="上下文知识")
        search_kb_range = gr.Dropdown(choices=article_list, label='检索范围', interactive=True)
        return selectable_knowledge_bases_checkbox_group, group_info_json[username][kb_embedding_type], book_type, search_kb_range, [], '', [], config_info

    def delete_article_group(self, db_type, embed_type, sel_kb_checkbox_grp, config_info):
        '''
        删除知识库
        :param db_type: 数据库类型
        :param embed_type: 编码类型
        :param sel_kb_checkbox_grp: 要删除的知识库列表
        :param config_info: 所有变量
        :return:
        '''
        article_group_path = config_info[db_type]['article_group']
        group_info_json = read_json_file(article_group_path)
        username = config_info['username']

        group_info_json = self.delete_group_entries(group_info_json, username, embed_type, sel_kb_checkbox_grp)
        save_json_file(group_info_json, article_group_path)
        article_list = list(group_info_json[username][embed_type].keys())
        config_info['article_group_dict'] = group_info_json
        sel_kb_chk_grp = gr.CheckboxGroup(choices=article_list, label='已有知识库', interactive=True)
        book_type = gr.Dropdown(choices=article_list, label="上下文知识")
        search_kb_range = gr.Dropdown(choices=article_list, label='检索范围', interactive=True)
        return sel_kb_chk_grp, group_info_json[username][embed_type], book_type, search_kb_range, config_info

    def select_article_group(self, search_database_type, search_embedding_type, search_kb_range, search_tok_k, search_text, config_info):
        '''
        根据条件进行搜索
        :param search_database_type: 数据库
        :param search_embedding_type: 编码方式
        :param search_kb_range: 知识库名字
        :param search_tok_k: 返回条数
        :param search_text: 搜索内容
        :param config_info: 全部配置
        :return:
        '''
        df = self.db_class.select_df_to_db(search_database_type, search_embedding_type, search_kb_range, search_tok_k, search_text, config_info)
        return df, config_info

class DealChat():
    def __init__(self):
        self.parse_class = ParseFileType()
        self.db_class = DealDataToDB()
        self.deal_rag_class = DealRag()

    def deal_history(self, rag_str, history, system_input, limit):
        '''保留多少条聊天记录'''
        message_list = []
        for recode in history:
            if recode['role'] == 'user':
                message_list.append({'role':'user', 'content':recode['content']})
            elif recode['role'] == 'assistant':
                if len(recode['metadata']) == 0:
                    message_list.append({'role':'assistant', 'content':recode['content']})
        if len(message_list) > limit:
            message_list = message_list[-limit:]
        system_set = {"role": "system", "content":f"{system_input}"}
        message_list.insert(0, system_set)
        message_list.append({'role':'user', 'content':f'{rag_str}'})
        return message_list

    def chat(self, message, history, model_type, chat_embedding_type, chat_database_type, book_type, is_connected_network, system_input, config_info):
        '''
        处理用户聊天
        :param message: 用户的输入信息
        :param history: 上下文
        :param model_type: 模型类型
        :param chat_embedding_type: 编码方式
        :param chat_database_type:  数据库
        :param book_type:  上下文内容
        :param is_connected_network: 是否联网
        :param system_input: 角色设置
        :param config_info: 全局配置
        :return:
        '''
        upload_file = message['files']
        user_ask = message['text']

        if len(upload_file) != 0:
            result_list = self.parse_class.parse_many_files(upload_file, chat_database_type, chat_embedding_type, config_info)
            config_info = self.db_class.save_df_to_db(chat_database_type, chat_embedding_type, result_list, True, 'temp', config_info)
            group_article_all = read_json_file(config_info[chat_database_type]['article_group'])
            username = config_info['username']
            if username not in group_article_all:
                group_article_all[username] = {}
            if chat_embedding_type not in group_article_all[username]:
                group_article_all[username][chat_embedding_type] = {}
            group_article = group_article_all[username][chat_embedding_type]
            group_article['temp'] = ['temp']
            group_article['temp'].extend(group_article[book_type])
            self.deal_rag_class.add_article_group(chat_database_type, chat_embedding_type, group_article['temp'],
                              'temp', config_info)
        df = self.db_class.select_df_to_db(chat_database_type, chat_embedding_type, book_type,
                                       config_info['rag_top_k'], user_ask, config_info)
        df = df.drop_duplicates(subset=['content'])

        sleep_time = 0.1
        start_time = time.time()
        response = ChatMessage(
            content="",
            metadata={"title": "涉及的相关信息", "id": 0, "status": "pending"}
        )
        yield response

        thoughts = []
        for index, row in df.iterrows():
            if chat_database_type == 'lancedb':
                if row['score'] >0.35:
                    continue
            elif chat_database_type == 'milvus':
                if row['score'] < 0.65:
                    continue
            temp_str = ''
            temp_str += f'标题：{row["title"]}'
            temp_str += f'内容:{row["content"]}'
            temp_str += f'来源：{row["file_from"]}'
            thoughts.append(temp_str)

        accumulated_thoughts = ""
        for thought in thoughts:
            time.sleep(sleep_time)
            accumulated_thoughts += f"- {thought}\n\n"
            response.content = accumulated_thoughts.strip()
            yield response

        response.metadata["status"] = "done"
        response.metadata["duration"] = time.time() - start_time
        yield response

        rag_str = '\n'.join(thoughts)
        rag_str += f'以上是参考信息，不一定与用户要求相关，以下是用户的要求：{user_ask}'

        if model_type.lower().startswith('local_qwen'):
            model_class = QwenChatModel()
            model_class.init_model(config_info['local_model_name_path_dict'][model_type])

            input_messages = self.deal_history(rag_str, history, system_input, config_info['saved_chat_record_count'])
            streamer = model_class.model_stream_detect(input_messages)

            response_content = ChatMessage(
                content="",
            )
            yield response_content

            output_str = ''
            for new_text in streamer:  # 从 streamer 中逐段获取生成的文本
                output_str += new_text
                response_content.content = output_str.strip()
                yield response_content

            response = [
                response,
                response_content
            ]
            yield response
        elif model_type.lower().startswith('ollama_'):
            emb_model_class = OllamaClient(config_info['ollama']['host'], config_info['ollama']['port'])
