import os
import gradio as gr
from local.chat_model.chat_do import local_chat

from ocr.ocr_model_select import get_result_image
from config import conf_yaml
from local.MiniCPM.minicpm_vl_detect.chat import minicpm_ui
from local.ui.adult_ui import adult_ui_show
from utils.plot_data import create_pie_chart
from local.rag.deal_many_file import deal_mang_knowledge_files, add_group_database, delete_group_database, delete_article_from_database

from utils.tool import read_user_info_dict, read_md_doc
from utils.config_init import get_database_config, chat_model_dict, user_password_info_dict_path, database_type, tmp_dir_path
from local.user.user_auth_management import AuthManager

default_system = conf_yaml['ui_conf']['default_system']
database_dir, articles_user_path, kb_article_map_path = get_database_config()
os.environ["GRADIO_TEMP_DIR"] = tmp_dir_path

def clear_session():
    return '', [], []

def get_username(request: gr.Request):
    article_dict = read_user_info_dict(request.username, articles_user_path)       # id:value
    kb_article_dict = read_user_info_dict(request.username, kb_article_map_path)
    return article_dict, kb_article_dict

with gr.Blocks() as demo:
    article_dict_state = gr.JSON({}, visible=False)
    kb_article_dict_state = gr.JSON({}, visible=False)
    demo.load(get_username, None, [article_dict_state, kb_article_dict_state])
    with gr.Tabs():
        with gr.TabItem("聊天机器人"):
            with gr.TabItem("本地文字"):
                default_history = []
                history_state = gr.State(value=default_history)
                with gr.Row():
                    model_type = gr.Dropdown(list(chat_model_dict.keys()), label="model type")
                    model_name = gr.Dropdown([], label="model name")
                    steam_check_box = gr.CheckboxGroup(["流式输出"], label="输出形式")

                @model_type.change(inputs=model_type, outputs=model_name)
                def update_cities(model_type):
                    model_list = list(chat_model_dict[model_type])
                    return gr.Dropdown(choices=model_list, value=model_list[0], interactive=True)


                with gr.Row():
                    with gr.Column(scale=3):
                        system_input = gr.Textbox(value=default_system, lines=1, label='角色设置')

                chatbot = gr.Chatbot(label='qwen2', show_copy_button=True)
                with gr.Row():
                    with gr.Column(scale=1):
                        book_type = gr.Dropdown(choices=[], label="上下文知识")
                    with gr.Column(scale=4):
                        textbox = gr.Textbox(lines=1, label='输入')

                with gr.Row():
                    clear_history = gr.Button("🧹 清除历史")
                    sumbit = gr.Button("🚀 发送")

                textbox.submit(local_chat,
                               inputs=[textbox, chatbot, system_input, history_state, model_type, model_name,
                                       steam_check_box, book_type],
                               outputs=[textbox, chatbot, history_state])

                sumbit.click(local_chat,
                             inputs=[textbox, chatbot, system_input, history_state, model_type, model_name,
                                     steam_check_box, book_type],
                             outputs=[textbox, chatbot, history_state])

                clear_history.click(fn=clear_session,
                                    inputs=[],
                                    outputs=[textbox, chatbot, history_state])
            with gr.TabItem("本地图片"):
                minicpm_ui()

            with gr.TabItem('gpu'):
                gpu_button = gr.Button('刷新')
                gpu_plot = gr.Plot(label="forecast", format="png")
                gpu_button.click(create_pie_chart, inputs=None, outputs=gpu_plot)

            with gr.TabItem('rag'):
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
                rag_submit_files_button.click(
                    deal_mang_knowledge_files,
                    inputs=[rag_upload_file, is_same_group, knowledge_name],
                    outputs=[article_dict_state, is_same_group, knowledge_name, rag_upload_file]
                )

            with gr.TabItem('知识库'):
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
                # 删除知识库按钮点击事件
                delete_knowledge_base_button.click(
                    delete_group_database,
                    inputs=[selectable_knowledge_bases_checkbox_group],
                    outputs=[selectable_knowledge_bases_checkbox_group, knowledge_base_info_json_table,
                             kb_article_dict_state]
                )
                rag_delete_button.click(
                    delete_article_from_database,
                    inputs=[rag_checkboxgroup, article_dict_state],
                    outputs=[article_dict_state, knowledge_base_info_json_table]
                )

                @knowledge_base_info_json_table.change(inputs=knowledge_base_info_json_table,
                                                       outputs=[selectable_knowledge_bases_checkbox_group, book_type])
                def update_selectable_knowledge_bases(knowledge_base_info):
                    return gr.CheckboxGroup(choices=list(knowledge_base_info.keys()), label='可选知识库'),gr.Dropdown(choices=list(knowledge_base_info.keys()), label="上下文知识")

                @article_dict_state.change(inputs=article_dict_state,
                                                       outputs=[rag_checkboxgroup, selectable_documents_checkbox_group])
                def update_selectable_knowledge_bases(input_value):
                    return gr.CheckboxGroup(choices=list(input_value.values()), label="rag管理"), gr.CheckboxGroup(choices=list(input_value.values()), label='可选文章')

                @kb_article_dict_state.change(inputs=kb_article_dict_state, outputs=[selectable_knowledge_bases_checkbox_group, book_type, knowledge_base_info_json_table])
                def update_selectable_knowledge_bases_checkbox_group_and_book_type(input_value):
                    return gr.CheckboxGroup(choices=list(input_value.keys()), label='已有知识库'), gr.Dropdown(choices=list(input_value.keys()), label="上下文知识"), input_value
            with gr.TabItem('ToDo'):
                need_to_do_string = read_md_doc('./readme.md')
                gr.Markdown(need_to_do_string)

        with gr.TabItem("本地视频播放"):
            adult_ui_show()

        with gr.TabItem('ocr识别'):
            with gr.Row():
                ocr_model_type = gr.Dropdown(['StepfunOcr'], value='StepfunOcr',label="选择模型",info='ocr在线识别')
            with gr.Row():
                img_input = gr.Image(type='filepath', label='Image')
                img_output = gr.Image()
                ocr_text = gr.Textbox(lines=20, max_lines=50, label='ocr_ouput', interactive=True, show_copy_button=True, container=True)
            img_input.upload(get_result_image, inputs=[img_input, ocr_model_type], outputs=[img_input, img_output, ocr_text])

# demo.launch(server_name='0.0.0.0')

# auth_manager = AuthManager(user_password_info_dict_path)
# demo.launch(server_name='0.0.0.0', auth=auth_manager.verify_auth)

demo.launch(server_name='0.0.0.0', auth=[('a', "a"), ('b', 'b'), ('pandas', '123')], server_port=7680)