import copy
import os

import lancedb
import ollama
import gradio as gr
import pandas as pd
import hashlib
from utils.tool import singleton, generate_unique_id, read_json_file, save_json_file, reverse_dict
from local.model.emb_model.bge_m3 import EmbBgeM3
from local.model.ollama_model.ollama_func import OllamaClient
from local.database.lancedb.data_to_lancedb import create_or_add_data_to_lancedb

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

class ParseFileType():
    def __init__(self):
        self.emb_class = Deal_emb()

    def parse_csv(self):
        pass

    def parse_docx(self):
        pass

    def parse_image(self):
        pass

    def parse_markdown(self):
        pass

    def parse_pdf(self, file_path, user_id, database_type, embedding_class, config_info):
        from local.rag.parse.pdf_parse import parse_pdf_do
        df = parse_pdf_do(file_path, user_id, database_type, embedding_class, config_info)
        return df

    def parse_video(self):
        pass

    def parse_voice(self):
        pass

    def parse_xlsx(self):
        pass

    def parse_single_file(self):
        pass

    def parse_many_files(self, files_path, database_type, embedding_type, config_info):
        #todo因为uuid的原因，导致hash失去效果了
        self.emb_model_class = self.emb_class.load_model(embedding_type, config_info)
        user_id = config_info['user_id']
        result_list = []
        for file_path in files_path:
            filname, suffix = os.path.splitext(os.path.basename(file_path))
            if suffix == '.pdf':
                df = self.parse_pdf(file_path, user_id, database_type, self.emb_model_class, config_info)
            elif suffix == '.csv':
                df = self.parse_csv()
            elif suffix == '.xlsx':
                df = self.parse_xlsx()
            elif suffix == '.md':
                df = self.parse_markdown()
            elif suffix == '.docx':
                df = self.parse_docx()
            elif suffix in ['.jpg', '.jpeg', '.png']:
                df = self.parse_image()
            elif suffix == '.mp3':
                df = self.parse_voice()
            elif suffix == '.mp4':
                df = self.parse_video()
            else:
                gr.Warning(f'{os.path.basename(file_path)}不支持解析')
                continue
            result_list.append(df)
        return result_list

class DealDataToDB():
    def __init__(self):
        self.emb_class = Deal_emb()

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

    def save_df_to_mysql(self, df_list):
        pass

    def save_df_to_milvus(self, df_list):
        pass

    def save_df_to_es(self, df_list):
        pass

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

    def save_df_to_db(self, rag_database_type, rag_embedding_type, info_list, is_same_group, article_name, config_info):
        if rag_database_type == 'lancedb':
            config_info = self.save_df_to_lancedb(rag_embedding_type, info_list, is_same_group, article_name, config_info)
        return config_info

    def delete_df_to_lancedb(self, rag_database_type, rag_embedding_type, rag_checkboxgroup, config_info):
        '''从lancedb中删除数据'''
        data_dir = config_info[rag_database_type]['data_dir']
        id2article_path = config_info[rag_database_type]['id2article']
        db = lancedb.connect(data_dir)
        id2article_info = read_json_file(id2article_path)
        username = config_info['username']
        if username not in id2article_info:
            id2article_info[username] = {}
        if rag_embedding_type not in id2article_info[username]:
            id2article_info[username][rag_embedding_type] = {}

        id2article_dict = id2article_info[username][rag_embedding_type]
        article2id_dict = reverse_dict(id2article_dict)

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
        config_info['article_group_dict'] = group_articles_dict_copy
        return config_info

    def delete_df_to_db(self, rag_database_type, rag_embedding_type, rag_checkboxgroup, config_info):
        if rag_database_type == 'lancedb':
            config_info = self.delete_df_to_lancedb(rag_database_type, rag_embedding_type, rag_checkboxgroup, config_info)
            return config_info

    def select_df_to_lancedb(self, search_database_type, search_kb_range, search_tok_k, search_text, config_info):
        emb_model_name = self.embedding_class.model_name
        vector = self.embedding_class.parse_single_sentence(
            model_name=emb_model_name,
            sentence=search_text
        )[0]
        data_dir = config_info[search_database_type]['data_dir']
        id2article_dict = config_info['id2article_dict']
        article2id_dict = reverse_dict(id2article_dict)
        article_group_dict = config_info['article_group_dict']
        db = lancedb.connect(data_dir)
        records_df = pd.DataFrame()
        search_tok_k = int(search_tok_k)
        for article_name in article_group_dict[search_kb_range]:
            tb = db.open_table(article2id_dict[article_name])
            records = tb.search(vector).limit(search_tok_k).to_pandas()
            records_df = pd.concat((records, records_df))
        sorted_records_df = records_df.sort_values(by='_distance').iloc[:search_tok_k]
        result_list = []
        for _, row in sorted_records_df.iterrows():
            info = {}
            info['title'] = row['title']
            info['content'] = row['content']
            info['file_from'] = row['file_from']
            result_list.append(info)
        return pd.DataFrame(result_list)

    def select_df_to_db(self, search_database_type, search_embedding_type, search_kb_range, search_tok_k, search_text, config_info):
        self.embedding_class = self.emb_class.load_model(search_embedding_type, config_info)
        if search_database_type=='lancedb':
            df = self.select_df_to_lancedb(search_database_type, search_kb_range, search_tok_k, search_text, config_info)
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
            rag_checkboxgroup = gr.CheckboxGroup(choices=[], label="rag列表", interactive=True)
            book_type = gr.Dropdown(choices=[], label="上下文知识")
            selectable_documents_checkbox_group = gr.CheckboxGroup(choices=[], label='可选文章', interactive=True)
            selectable_knowledge_bases_checkbox_group = gr.CheckboxGroup(choices=[], label='已有知识库', interactive=True)
            search_kb_range = gr.Dropdown(choices=[], label='检索范围', interactive=True)
        else:
            article_list = list(config_info['id2article_dict'].values())
            group_list = list(config_info['article_group_dict'].keys())
            rag_checkboxgroup = gr.CheckboxGroup(choices=article_list, label="rag列表", interactive=True)
            book_type = gr.Dropdown(choices=group_list, label="上下文知识")
            selectable_documents_checkbox_group = gr.CheckboxGroup(choices=article_list, label='可选文章', interactive=True)
            selectable_knowledge_bases_checkbox_group = gr.CheckboxGroup(choices=group_list, label='已有知识库',
                                                                         interactive=True)
            search_kb_range = gr.Dropdown(choices=group_list, label='检索范围', interactive=True)
        article_group_dict = config_info['article_group_dict']
        return rag_checkboxgroup, book_type, selectable_documents_checkbox_group, selectable_knowledge_bases_checkbox_group, search_kb_range, article_group_dict, config_info

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
        article_list = group_info_json[username][embed_type].keys()
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
        pass

    def chat(self):
        pass