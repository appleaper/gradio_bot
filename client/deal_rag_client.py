import json
import requests
import gradio as gr
import pandas as pd
global_port = '4500'
global_ip = '127.0.0.1'

class DealRagClient():

    def delete_article_client(
            self,
            rag_database_type: str,
            rag_embedding_type: str,
            rag_checkboxgroup: list,
            config_info: dict
    ):
        '''
        删除文章
        :param rag_database_type: 数据库类型
        :param rag_embedding_type: 编码类型
        :param rag_checkboxgroup: 选择的文章
        :param config_info: 配置信息
        :return:
        '''
        data = {
            'rag_database_type': rag_database_type,
            'rag_embedding_type': rag_embedding_type,
            'rag_checkboxgroup': rag_checkboxgroup,
            'config_info': config_info
        }
        response = requests.post(f'http://{global_ip}:{global_port}/delete_article', json=data)
        print(response)
        json_dict = response.json()
        config_info = json_dict['config_info']
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

    def add_article_client(
            self,
            rag_database_type: str,
            rag_embedding_type: str,
            is_same_group: str,
            knowledge_name: str,
            rag_upload_file: list,
            config_info: dict
    ):
        data = {
            'rag_database_type': rag_database_type,
            'rag_embedding_type': rag_embedding_type,
            'is_same_group': is_same_group,
            'knowledge_name': knowledge_name,
            'rag_upload_file': rag_upload_file,
            'config_info': config_info
        }
        response = requests.post(f'http://{global_ip}:{global_port}/add_article', json=data)
        json_dict = response.json()
        config_info = json_dict['config_info']

        article_list = list(config_info['id2article_dict'].values())
        rag_checkboxgroup = gr.CheckboxGroup(choices=article_list, label="rag列表", interactive=True)
        selectable_documents_checkbox_group = gr.CheckboxGroup(choices=article_list, label='可选文章', interactive=True)
        config_json = gr.JSON(value=config_info,visible=False)
        return rag_checkboxgroup, selectable_documents_checkbox_group, '', '', [], config_json

    def add_article_group_client(
            self,
            kb_database_type: str,
            kb_embedding_type: str,
            selectable_documents_checkbox_group: list,
            new_kb_name_textbox: str,
            config_info: dict
    ):
        data = {
            'kb_database_type': kb_database_type,
            'kb_embedding_type': kb_embedding_type,
            'selectable_documents_checkbox_group': selectable_documents_checkbox_group,
            'new_kb_name_textbox': new_kb_name_textbox,
            'config_info': config_info,
        }
        response = requests.post(f'http://{global_ip}:{global_port}/add_article_group', json=data)
        json_dict = response.json()
        group_info_json = json_dict['group_info_json']
        username = json_dict['username']
        article_list = group_info_json[username][kb_embedding_type].keys()
        config_info['article_group_dict'] = group_info_json

        selectable_knowledge_bases_checkbox_group = gr.CheckboxGroup(choices=article_list, label='已有知识库',
                                                                     interactive=True)
        book_type = gr.Dropdown(choices=article_list, label="上下文知识")
        search_kb_range = gr.Dropdown(choices=article_list, label='检索范围', interactive=True)
        return selectable_knowledge_bases_checkbox_group, group_info_json[username][
            kb_embedding_type], book_type, search_kb_range, [], '', [], config_info


    def delete_article_group_client(
            self,
            kb_database_type: str,
            kb_embedding_type: str,
            selectable_knowledge_bases_checkbox_group: list,
            config_info: dict
    ):
        # kb_database_type, kb_embedding_type, selectable_knowledge_bases_checkbox_group, config_info
        data = {
            'kb_database_type': kb_database_type,
            'kb_embedding_type': kb_embedding_type,
            'selectable_knowledge_bases_checkbox_group': selectable_knowledge_bases_checkbox_group,
            'config_info': config_info,
        }
        response = requests.post(f'http://{global_ip}:{global_port}/delete_article_group', json=data)
        json_dict = response.json()
        group_info_json = json_dict['group_info_json']
        username = json_dict['username']

        article_list = list(group_info_json[username][kb_embedding_type].keys())
        config_info['article_group_dict'] = group_info_json
        sel_kb_chk_grp = gr.CheckboxGroup(choices=article_list, label='已有知识库', interactive=True)
        book_type = gr.Dropdown(choices=article_list, label="上下文知识")
        search_kb_range = gr.Dropdown(choices=article_list, label='检索范围', interactive=True)
        return sel_kb_chk_grp, group_info_json[username][kb_embedding_type], book_type, search_kb_range, config_info

    def select_article_group(
            self,
            search_database_type: str,
            search_embedding_type: str,
            search_kb_range: str,
            search_tok_k: int,
            search_text: str,
            config_info: dict
    ):
        data = {
            'search_database_type': search_database_type,
            'search_embedding_type': search_embedding_type,
            'search_kb_range': search_kb_range,
            'search_tok_k': search_tok_k,
            'search_text': search_text,
            'config_info': config_info,
        }
        response = requests.post(f'http://{global_ip}:{global_port}/select_article_group', json=data)
        json_dict = response.json()
        df_json = json_dict['df_json']
        config_info = json_dict['config_info']
        df = pd.DataFrame(json.loads(df_json))
        return df, config_info