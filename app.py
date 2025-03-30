import os
import gradio as gr
import pandas as pd
from config import conf_yaml
from utils.tool import read_md_doc
from local.ui.ocr_ui import ocr_ui_show
from local.ui.adult_ui import adult_ui_show
from local.ui.voice_ui import voice_ui_show
from utils.plot_data import create_pie_chart
from local.chat_model.chat_do import local_chat
from utils.config_init import akb_conf_class, chat_model_dict, tmp_dir_path
from local.rag.search_data_from_database import search_data_from_database_do, get_user_select_info
from local.rag.deal_many_file import deal_mang_knowledge_files, add_group_database, delete_group_database, delete_article_from_database

parm2chinese = {
    '本地聊天':{
        '变量类': {
            'chat_model_dict': '可供选择的模型',
            'default_system': '默认提示词',
        },
        '组件类': {
            'article_dict_state': '文章id和文章名字的映射关系',
            'kb_article_dict_state': '知识库内有什么文章',
            'model_type': '模型厂商',
            'model_name': '模型具体型号',
            'chat_database_type': '聊天关联的数据库',
            'is_connected_network': '是否联网',
            'system_input': '角色设置',
            'chatbot': '聊天界面'
        }
    },
    'gpu使用查询':{
        '组件':{
            'gpu_button': '查询GPU使用情况',
            'gpu_plot': 'GPU使用的饼状图'
        }
    },
    'rag文章管理':{
        'rag_database_type': 'rag关联的数据库',
        'rag_checkboxgroup':'rag文章列表',
        'rag_delete_button':'删除rag文章',
        'is_same_group':'上传的文章是否视为同一组',
        'knowledge_name':'文章名字',
        'rag_upload_file':'rag文章上传按钮',
        'rag_submit_files_button':'rag文章解析按钮'
    },
    '知识库管理':{
        'kb_database_type':'知识库关联的数据库',
        'selectable_documents_checkbox_group':'组成一个知识库的可选文章',
        'selectable_knowledge_bases_checkbox_group':'以有知识库',
        'new_knowledge_base_name_textbox':'新建的知识库名字',
        'create_knowledge_base_button':'新建知识库按钮',
        'delete_knowledge_base_button':'删除知识库按钮',
        'knowledge_base_info_json_table':'展示知识库内有什么文章'
    },
    '搜索':{
        'search_database_type':'搜索关联的数据库',
        'search_kb_range': '检索的知识库',
        'search_tok_k':'返回的检索结果数量',
        'search_text':'你想搜索的内容',
        'search_button':'搜索按钮',
        'search_show':'搜索结果表格化展示',
        'search_info_title':'搜索结果的标题',
        'search_info_file_from': '搜索结果的来源',
        'search_info_content':'搜索结果的正文',
    }
}

default_system = conf_yaml['ui_conf']['default_system']
gr.set_static_paths(paths=[tmp_dir_path])

def test(message, history):
    return "test"

with gr.Blocks() as demo:
    article_dict_state = gr.JSON({}, visible=False)
    kb_article_dict_state = gr.JSON({}, visible=False)
    demo.load(akb_conf_class.get_username, None, [article_dict_state, kb_article_dict_state])

    with gr.Tabs():
        with gr.TabItem("聊天机器人"):
            with gr.TabItem("本地文字"):
                with gr.Row():
                    model_type = gr.Dropdown(choices=list(chat_model_dict.keys()), value='',label="模型厂商", interactive=True)
                    model_name = gr.Dropdown([], label="模型具体型号")
                    chat_database_type = gr.Dropdown(choices=['lancedb', 'milvus', 'mysql', 'es'], label='关联的数据库', interactive=True)
                with gr.Row():
                    is_connected_network = gr.Checkbox(label='联网搜索', value=False, scale=1, info='选择是否联网，默认不联网')
                    book_type = gr.Dropdown(choices=[], label="上下文知识")
                    system_input = gr.Textbox(value=default_system, lines=1, label='角色设置', scale=6)
                with gr.Row():
                    chatbot = gr.ChatInterface(
                        local_chat,
                        additional_inputs=[system_input, model_type, model_name,
                                       book_type, chat_database_type, is_connected_network],
                        multimodal=True,
                        type="messages",    # 控制多媒体上传
                        editable=True,      # 用户可以编辑过去的消息重新生成
                        save_history=True,  # 聊天记录保存到本地
                        stop_btn=True,      # 停止生成
                        autoscroll=True,    # 生成时自动滚动到底部
                    )

            with gr.TabItem('gpu'):
                gpu_button = gr.Button('刷新')
                gpu_plot = gr.Plot(label="forecast", format="png")
                gpu_button.click(create_pie_chart, inputs=None, outputs=gpu_plot)

            with gr.TabItem('rag'):
                with gr.Row():
                    rag_database_type = gr.Dropdown(choices=['lancedb', 'milvus', 'mysql', 'es'], label='关联的数据库', value='lancedb', interactive=True)
                with gr.Row():
                    rag_checkboxgroup = gr.CheckboxGroup(choices=[], label="rag列表")
                with gr.Row():
                    with gr.Column(scale=1):
                        rag_delete_button = gr.Button(value='删除')
                with gr.Row():
                    with gr.Column(scale=1):
                        is_same_group = gr.Radio(["是", "否"], label="是否为同一个组", info='若选是，则多个文章视为一篇文章，否则将视为多篇文章')
                    with gr.Column(scale=4):
                        knowledge_name = gr.Textbox(lines=1, label='文章名字',
                                                    placeholder='给文章起个名字吧，不起的话，默认为上传的第一个文件的名字')
                rag_upload_file = gr.Files(label='上传文件，支持pdf,csv,md,jpg,png,jpeg,xlsx,,docx,mp4,mp3格式，上传csv必须含title和content这两列。而且代码只解析这两列。')
                rag_submit_files_button = gr.Button(value='开始解析')

            with gr.TabItem('知识库'):
                with gr.Row():
                    kb_database_type = gr.Dropdown(choices=['lancedb', 'milvus', 'mysql', 'es'], label='关联的数据库', value='lancedb', interactive=True)
                # 可选文章复选框组
                selectable_documents_checkbox_group = gr.CheckboxGroup(
                    choices=[], label='可选文章'
                )
                # 可选知识库复选框组
                selectable_knowledge_bases_checkbox_group = gr.CheckboxGroup(
                    choices=[], label='已有知识库')
                # 新建知识库名输入文本框
                new_knowledge_base_name_textbox = gr.Textbox(lines=1, label='知识库名', placeholder='给新建的知识库起个名字吧')
                # 新建知识库按钮
                create_knowledge_base_button = gr.Button("新建知识库")
                # 删除知识库按钮
                delete_knowledge_base_button = gr.Button("删除知识库")
                # 显示知识库信息的 JSON 表格
                knowledge_base_info_json_table = gr.JSON()

                create_knowledge_base_button.click(
                    add_group_database,
                    inputs=[selectable_documents_checkbox_group, new_knowledge_base_name_textbox],
                    outputs=[selectable_documents_checkbox_group, new_knowledge_base_name_textbox,
                             knowledge_base_info_json_table, kb_article_dict_state]
                )

            with gr.TabItem('搜索'):
                with gr.Row():
                    search_database_type = gr.Dropdown(choices=['lancedb', 'milvus', 'mysql', 'es'], label='关联的数据库', value='lancedb', interactive=True)
                    search_kb_range = gr.Dropdown(choices=[], label='检索范围')
                    search_tok_k = gr.Textbox(value='3', label='返回多少条结果')
                with gr.Row():
                    search_text = gr.Textbox(placeholder='输入你想搜索的内容',scale=4)
                    search_button = gr.Button(value='搜索', scale=1)
                with gr.Row():
                    search_show = gr.DataFrame(value=pd.DataFrame([]))

                with gr.Row():
                    search_info_title = gr.Markdown(label='搜索结果标题')
                    search_info_file_from = gr.Markdown(label='搜索结果来源')
                with gr.Row():
                    search_info_content = gr.Markdown(label='搜索结果正文')

            with gr.TabItem('ToDo'):
                need_to_do_string = read_md_doc('计划和目前的bug.md')
                gr.Markdown(need_to_do_string)
        with gr.TabItem("本地视频播放"):
            adult_ui_show()

        with gr.TabItem('ocr识别'):
            ocr_ui_show()

        with gr.TabItem('语音识别'):
            voice_ui_show()



    @model_type.change(inputs=model_type, outputs=model_name)
    def update_cities(model_type):
        model_list = list(chat_model_dict[model_type])
        return gr.Dropdown(choices=model_list, value=model_list[0], interactive=True)


    rag_submit_files_button.click(
        deal_mang_knowledge_files,
        inputs=[rag_upload_file, is_same_group, knowledge_name, rag_database_type],
        outputs=[article_dict_state, is_same_group, knowledge_name, rag_checkboxgroup]
    )


    @article_dict_state.change(inputs=article_dict_state,
                               outputs=[rag_checkboxgroup, selectable_documents_checkbox_group])
    def update_selectable_knowledge_bases(input_value):
        return gr.CheckboxGroup(choices=list(input_value.values()), label="rag管理"), gr.CheckboxGroup(
            choices=list(input_value.values()), label='可选文章')


    search_button.click(
        search_data_from_database_do,
        inputs=[search_database_type, search_text, search_kb_range, search_tok_k],
        outputs=[search_show]
    )
    search_text.submit(
        search_data_from_database_do,
        inputs=[search_database_type, search_text, search_kb_range, search_tok_k],
        outputs=[search_show]
    )
    search_show.select(
        get_user_select_info,
        inputs=None,
        outputs=[search_info_title, search_info_content, search_info_file_from]
    )


    @chat_database_type.change(
        inputs=chat_database_type,
        outputs=[rag_database_type, kb_database_type, search_database_type, article_dict_state, kb_article_dict_state])
    def chat_database_type_change_1(input, request: gr.Request):
        return akb_conf_class.database_type_dropdowns(input, request)


    @rag_database_type.change(
        inputs=rag_database_type,
        outputs=[chat_database_type, kb_database_type, search_database_type, article_dict_state, kb_article_dict_state])
    def chat_database_type_change_2(input, request: gr.Request):
        return akb_conf_class.database_type_dropdowns(input, request)


    @kb_database_type.change(
        inputs=kb_database_type,
        outputs=[chat_database_type, rag_database_type, search_database_type, article_dict_state,
                 kb_article_dict_state])
    def chat_database_type_change_3(input, request: gr.Request):
        return akb_conf_class.database_type_dropdowns(input, request)


    @search_database_type.change(
        inputs=search_database_type,
        outputs=[chat_database_type, rag_database_type, kb_database_type, article_dict_state, kb_article_dict_state])
    def chat_database_type_change_4(input, request: gr.Request):
        return akb_conf_class.database_type_dropdowns(input, request)


    # rag删除文章
    rag_delete_button.click(
        delete_article_from_database,
        inputs=[rag_checkboxgroup, article_dict_state, rag_database_type],
        outputs=[article_dict_state, knowledge_base_info_json_table]
    )

    # 删除知识库按钮点击事件
    delete_knowledge_base_button.click(
        delete_group_database,
        inputs=[selectable_knowledge_bases_checkbox_group, kb_database_type],
        outputs=[selectable_knowledge_bases_checkbox_group, knowledge_base_info_json_table,
                 kb_article_dict_state]
    )


    # 知识库更新时触发
    @knowledge_base_info_json_table.change(inputs=knowledge_base_info_json_table,
                                           outputs=[selectable_knowledge_bases_checkbox_group, book_type,
                                                    search_kb_range])
    def update_selectable_knowledge_bases(knowledge_base_info):
        knowledge_name_list = list(knowledge_base_info.keys())
        a = gr.CheckboxGroup(choices=knowledge_name_list, label='可选知识库')
        b = gr.Dropdown(choices=knowledge_name_list, label="上下文知识")
        c = gr.Dropdown(choices=knowledge_name_list, label='检索范围')
        return a, b, c


    @kb_article_dict_state.change(
        inputs=kb_article_dict_state,
        outputs=[
            selectable_knowledge_bases_checkbox_group,
            book_type,
            knowledge_base_info_json_table,
            search_kb_range
        ])
    def update_selectable_knowledge_bases_checkbox_group_and_book_type(input_value):
        a = gr.CheckboxGroup(choices=list(input_value.keys()), label='已有知识库')
        b = gr.Dropdown(choices=list(input_value.keys()), label="上下文知识")
        c = input_value
        d = gr.Dropdown(choices=list(input_value.keys()), label="检索范围")
        return a, b, c, d

# demo.launch(server_name='0.0.0.0')
demo.launch(
    server_name='0.0.0.0',
    auth=[('a', "a"), ('b', 'b'), ('pandas', '123')],
    server_port=7680,
    allowed_paths=[r'D:\video']
)