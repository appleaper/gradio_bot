import pandas as pd
import gradio as gr
from utils.tool import singleton
from utils.plot_data import create_pie_chart
from utils.config_init import conf_class
from local.rag.deal_rag import DealRag, DealChat

rag_class = DealRag()
chat_class = DealChat()

@singleton
class UiDataDeal():
    def __init__(self):
        pass

    def init_default_value_ui(self, request:gr.Request):
        '''初始阿虎组件的初始值'''
        conf_class.init_config(request.username)
        conf_class.config_dict['username'] = request.username
        self.config_dict = conf_class.config_dict
        id2article_dict, article_group_dict = conf_class.get_user_id2article_and_article_group(
            self.config_dict['default_database_type'],
            self.config_dict['username'],
            self.config_dict['default_embed_model']
        )
        self.config_dict['id2article_dict'] = id2article_dict
        self.config_dict['article_group_dict'] = article_group_dict

        model_type = gr.Dropdown(choices=self.config_dict['all_model_list'], value=self.config_dict['default_chat_model'], label="模型厂商", interactive=True, filterable=True)
        encoding_type = gr.Dropdown(choices=self.config_dict['all_model_list'], value=self.config_dict['default_embed_model'],label='编码方式', interactive=True)
        database_type = gr.Dropdown(choices=self.config_dict['database_type'], value=self.config_dict['default_database_type'], label='关联的数据库', interactive=True)
        book_type = gr.Dropdown(choices=list(self.config_dict['article_group_dict'].keys()), value='空',label="上下文知识")
        chat_encoding_type = encoding_type
        chat_database_type = database_type
        rag_encoding_type = encoding_type
        rag_database_type = database_type
        kb_encoding_type = encoding_type
        kb_database_type = database_type
        system_input = gr.Textbox(value=self.config_dict['caht_default_system'], lines=1, label='角色设置', scale=6)
        rag_checkboxgroup = gr.CheckboxGroup(choices=list(self.config_dict['id2article_dict'].values()), label="rag列表", interactive=True)
        selectable_documents_checkbox_group = gr.CheckboxGroup(choices=list(self.config_dict['id2article_dict'].values()), label='可选文章', interactive=True)
        selectable_knowledge_bases_checkbox_group = gr.CheckboxGroup(
            choices=list(self.config_dict['article_group_dict'].keys()), label='已有知识库')
        knowledge_base_info_json_table = gr.JSON(value=self.config_dict['article_group_dict'])
        search_encoding_type = encoding_type
        search_database_type = database_type
        search_kb_range = gr.Dropdown(choices=list(self.config_dict['article_group_dict'].keys()), label='检索范围')
        return (model_type,chat_encoding_type,chat_database_type, book_type, system_input, rag_encoding_type,
                rag_database_type, rag_checkboxgroup, kb_encoding_type,
                kb_database_type, selectable_documents_checkbox_group,
                selectable_knowledge_bases_checkbox_group, knowledge_base_info_json_table,
                search_encoding_type, search_database_type, search_kb_range, self.config_dict)

    def find_different_element_position(self, lst):
        '''找出选出的那个选项'''
        element_count = {}
        for element in lst:
            if element in element_count:
                element_count[element] += 1
            else:
                element_count[element] = 1

        different_element = None
        for element, count in element_count.items():
            if count == 1:
                different_element = element
                break

        if different_element is not None:
            return lst.index(different_element)
        return 0

    def sync_variables(self,
                       chat_database_type, rag_database_type, kb_database_type, search_database_type,
                       chat_embedding_type, rag_embedding_type, kb_embedding_type, search_embedding_type
                       ):
        '''当切换数据库或者切换编码方式时，就切换对应的ui'''
        database_types = [chat_database_type, rag_database_type, kb_database_type, search_database_type]
        db_type_pos = self.find_different_element_position(database_types)

        chat_db = gr.Dropdown(choices=self.config_dict['database_type'], value=database_types[db_type_pos],label='关联的数据库', interactive=True)
        rag_db = gr.Dropdown(choices=self.config_dict['database_type'], value=database_types[db_type_pos], label='关联的数据库', interactive=True)
        kb_db = gr.Dropdown(choices=self.config_dict['database_type'], value=database_types[db_type_pos], label='关联的数据库', interactive=True)
        search_db = gr.Dropdown(choices=self.config_dict['database_type'], value=database_types[db_type_pos], label='关联的数据库', interactive=True)

        embedding_types = [chat_embedding_type, rag_embedding_type, kb_embedding_type, search_embedding_type]
        emb_type_pos = self.find_different_element_position(embedding_types)

        chat_emb_type = gr.Dropdown(choices=self.config_dict['all_model_list'], value=embedding_types[emb_type_pos],
                                    label='编码方式', interactive=True)
        rag_emb_type = gr.Dropdown(choices=self.config_dict['all_model_list'], value=embedding_types[emb_type_pos],
                                   label='编码方式', interactive=True)
        kb_emb_type = gr.Dropdown(choices=self.config_dict['all_model_list'], value=embedding_types[emb_type_pos],
                                  label='编码方式', interactive=True)
        search_emb_type = gr.Dropdown(choices=self.config_dict['all_model_list'],
                                      value=embedding_types[emb_type_pos], label='编码方式', interactive=True)

        id2article_dict, article_group_dict = conf_class.get_user_id2article_and_article_group(
            database_types[db_type_pos],
            self.config_dict['username'],
            embedding_types[emb_type_pos]
        )
        article_passage = list(id2article_dict.values())
        article_group_name = list(article_group_dict.keys())

        book_type = gr.Dropdown(choices=article_group_name, label="上下文知识")
        rag_checkboxgroup = gr.CheckboxGroup(choices=article_passage, label="rag列表", interactive=True)
        selectable_documents_checkbox_group = gr.CheckboxGroup(choices=article_passage, label='可选文章', interactive=True)
        selectable_knowledge_bases_checkbox_group = gr.CheckboxGroup(choices=article_group_name, label='已有知识库')
        knowledge_base_info_json_table = gr.JSON(value=article_group_dict)
        search_kb_range = gr.Dropdown(choices=article_group_name, label='检索范围', interactive=True)


        return (chat_db, rag_db, kb_db, search_db, chat_emb_type, rag_emb_type, kb_emb_type, search_emb_type,
                book_type, rag_checkboxgroup, selectable_documents_checkbox_group,
                selectable_knowledge_bases_checkbox_group, knowledge_base_info_json_table, search_kb_range)

    def update_rag_checkboxgroup(self, config_info):
        return gr.CheckboxGroup(choices=list(config_info['id2article_dict'].values()), label="rag列表", interactive=True)


    def search_click_for_details(self, evt: gr.SelectData):
        '''搜素结果点击获取细节'''
        title = evt.row_value[0]
        content = evt.row_value[1]
        file_from = evt.row_value[2]
        return title, file_from, content

def chat_ui_show(demo):
    ui_class = UiDataDeal()
    with gr.TabItem('chatbot'):
        config_info = gr.JSON(visible=False)
        with gr.Row():
            model_type = gr.Dropdown(choices=[], value='', label="模型厂商", interactive=True, filterable=True)
            chat_embedding_type = gr.Dropdown(choices=[], label='编码方式', interactive=True)
            chat_database_type = gr.Dropdown(choices=[], label='关联的数据库', interactive=True)
            book_type = gr.Dropdown(choices=[], label="上下文知识")

        with gr.Row():
            is_connected_network = gr.Checkbox(label='联网搜索', value=False, info='选择是否联网，默认不联网')
            system_input = gr.Textbox(value='', lines=1, label='角色设置', scale=6)
        with gr.Row():
            chatbot = gr.ChatInterface(
                chat_class.chat,
                additional_inputs=[model_type, chat_embedding_type, chat_database_type, book_type, is_connected_network, system_input, config_info],
                multimodal=True,
                type="messages",  # 控制多媒体上传
                editable=True,  # 用户可以编辑过去的消息重新生成
                save_history=True,  # 聊天记录保存到本地
                stop_btn=True,  # 停止生成
                autoscroll=True,  # 生成时自动滚动到底部
            )

    with gr.TabItem('gpu'):
        gpu_button = gr.Button('刷新')
        gpu_plot = gr.Plot(label="forecast", format="png")

    with gr.TabItem('rag'):
        with gr.Row():
            rag_database_type = gr.Dropdown(choices=[], label='关联的数据库', interactive=True)
            rag_embedding_type = gr.Dropdown(choices=[], label='编码方式', interactive=True)
        with gr.Row():
            rag_checkboxgroup = gr.CheckboxGroup(choices=[], label="rag列表", interactive=True)
        with gr.Row():
            rag_delete_button = gr.Button(value='删除')
        with gr.Row():
            with gr.Column(scale=1):
                is_same_group = gr.Radio(["是", "否"], label="是否为同一个组",
                                         info='若选是，则多个文章视为一篇文章，否则将视为多篇文章')
            with gr.Column(scale=4):
                knowledge_name = gr.Textbox(lines=1, label='文章名字',
                                            placeholder='给文章起个名字吧，不起的话，默认为上传的第一个文件的名字')
        rag_upload_file = gr.Files(
            label='上传文件，支持pdf,csv,md,jpg,png,jpeg,xlsx,,docx,mp4,mp3格式，上传csv必须含title和content这两列。而且代码只解析这两列。')
        rag_submit_files_button = gr.Button(value='开始解析')

    with gr.TabItem('知识库'):
        with gr.Row():
            kb_database_type = gr.Dropdown(choices=[], label='关联的数据库', value='lancedb', interactive=True)
            kb_embedding_type = gr.Dropdown(choices=[], label='编码方式', interactive=True)
        # 可选文章复选框组
        selectable_documents_checkbox_group = gr.CheckboxGroup(choices=[], label='可选文章', interactive=True)
        # 可选知识库复选框组
        selectable_knowledge_bases_checkbox_group = gr.CheckboxGroup(choices=[], label='已有知识库', interactive=True)
        # 新建知识库名输入文本框
        new_kb_name_textbox = gr.Textbox(lines=1, label='知识库名', placeholder='给新建的知识库起个名字吧')
        # 新建知识库按钮
        create_kb_group_button = gr.Button("新建知识库")
        # 删除知识库按钮
        delete_kb_button = gr.Button("删除知识库")
        # 显示知识库信息的 JSON 表格
        knowledge_base_info_json_table = gr.JSON()

    with gr.TabItem('搜索'):
        with gr.Row():
            search_database_type = gr.Dropdown(choices=[], label='关联的数据库', interactive=True)
            search_embedding_type = gr.Dropdown(choices=[], label='编码方式', interactive=True)
            search_kb_range = gr.Dropdown(choices=[], label='检索范围', interactive=True)
            search_tok_k = gr.Number(value=3, label='返回多少条结果')
        with gr.Row():
            search_text = gr.Textbox(placeholder='输入你想搜索的内容', scale=4)
            search_button = gr.Button(value='搜索', scale=1)
        with gr.Row():
            search_show = gr.DataFrame(value=pd.DataFrame([]))

        with gr.Row():
            search_info_title = gr.Markdown(label='搜索结果标题')
            search_info_file_from = gr.Markdown(label='搜索结果来源')
        with gr.Row():
            search_info_content = gr.Markdown(label='搜索结果正文')

    demo.load(
        ui_class.init_default_value_ui,
        inputs=None,
        outputs=[model_type,chat_embedding_type,chat_database_type, book_type, system_input, rag_embedding_type,
                rag_database_type, rag_checkboxgroup, kb_embedding_type,
                kb_database_type, selectable_documents_checkbox_group,
                selectable_knowledge_bases_checkbox_group, knowledge_base_info_json_table,
                search_embedding_type, search_database_type, search_kb_range, config_info]
    )

    gr.on(
        triggers=[
            chat_database_type.change, rag_database_type.change, kb_database_type.change, search_database_type.change,
            chat_embedding_type.change, rag_embedding_type.change, kb_embedding_type.change, search_embedding_type.change
        ],
        fn=ui_class.sync_variables,
        inputs=[
            chat_database_type, rag_database_type, kb_database_type, search_database_type,
            chat_embedding_type, rag_embedding_type, kb_embedding_type, search_embedding_type,
        ],
        outputs=[
            chat_database_type, rag_database_type, kb_database_type, search_database_type,
            chat_embedding_type, rag_embedding_type, kb_embedding_type, search_embedding_type,
            book_type,rag_checkboxgroup, selectable_documents_checkbox_group, selectable_knowledge_bases_checkbox_group,
            knowledge_base_info_json_table, search_kb_range
        ]
    )

    # GPU占比展示
    gpu_button.click(create_pie_chart, inputs=config_info, outputs=gpu_plot)

    rag_delete_button.click(
        rag_class.delete_article,
        inputs=[rag_database_type, rag_embedding_type, rag_checkboxgroup, config_info],
        outputs=[
            rag_checkboxgroup, book_type,
            selectable_documents_checkbox_group, selectable_knowledge_bases_checkbox_group,
            search_kb_range, knowledge_base_info_json_table, config_info
        ]
    ).then(
        ui_class.update_rag_checkboxgroup,
        inputs = [config_info],
        outputs = [rag_checkboxgroup]
    )

    rag_submit_files_button.click(
        rag_class.add_article,
        inputs=[rag_database_type, rag_embedding_type, is_same_group, knowledge_name, rag_upload_file, config_info],
        outputs=[rag_checkboxgroup, selectable_documents_checkbox_group, is_same_group, knowledge_name, rag_upload_file, config_info]
    )

    create_kb_group_button.click(
        rag_class.add_article_group,
        inputs=[kb_database_type, kb_embedding_type, selectable_documents_checkbox_group, new_kb_name_textbox, config_info],
        outputs=[selectable_knowledge_bases_checkbox_group, knowledge_base_info_json_table, book_type, search_kb_range, selectable_documents_checkbox_group, new_kb_name_textbox, selectable_knowledge_bases_checkbox_group, config_info]
    )

    delete_kb_button.click(
        rag_class.delete_article_group,
        inputs=[kb_database_type, kb_embedding_type, selectable_knowledge_bases_checkbox_group, config_info],
        outputs=[selectable_knowledge_bases_checkbox_group, knowledge_base_info_json_table, book_type, search_kb_range, config_info]
    )

    gr.on(
        triggers=[search_text.submit, search_button.click],
        fn=rag_class.select_article_group,
        inputs=[search_database_type, search_embedding_type, search_kb_range, search_tok_k, search_text, config_info],
        outputs=[search_show, config_info]
    )

    search_show.select(
        ui_class.search_click_for_details,
        inputs=None,
        outputs=[search_info_title, search_info_file_from, search_info_content]
    )

if __name__ == '__main__':
    '''
    我现在的逻辑是配制有改动就保存到本地中。
    '''
    with gr.Blocks() as demo:
        chat_ui_show(demo)
    demo.launch(
        auth=[('a', "a"), ('b', 'b'), ('pandas', '123')],
        debug=True
    )