import requests
import gradio as gr
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
