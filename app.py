import os
import gradio as gr
import pandas as pd

from local.ui.ocr_ui import ocr_ui_show
from local.ui.adult_ui import adult_ui_show
from local.ui.voice_ui import voice_ui_show
from local.ui.chat_ui import chat_ui_show


from utils.config_init import conf_class
from local.rag.search_data_from_database import search_data_from_database_do, get_user_select_info

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


with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.TabItem("聊天机器人"):
            chat_ui_show(demo)

        with gr.TabItem("本地视频播放"):
            adult_ui_show()

        with gr.TabItem('ocr识别'):
            ocr_ui_show()

        with gr.TabItem('语音识别'):
            voice_ui_show()


# demo.launch(server_name='0.0.0.0')
demo.launch(
    server_name='0.0.0.0',
    auth=[('a', "a"), ('b', 'b'), ('pandas', '123')],
    server_port=7680,
    allowed_paths=[r'D:\video']
)