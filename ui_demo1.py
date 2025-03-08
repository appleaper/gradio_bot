import gradio as gr
from local.chat_model.chat_do import local_chat
from video.video_play import load_local_video, mark_video_like
from video.shutdown_computer import shutdown_computer
from video.cut_video import video_cut
from ocr.ocr_model_select import get_result_image
from config import conf_yaml
from local.MiniCPM.minicpm_vl_detect.chat import minicpm_ui
from utils.plot_data import create_pie_chart
from local.rag.deal_many_file import deal_mang_knowledge_files, drop_lancedb_table
from local.rag.util import read_rag_name_dict, read_md_doc
from utils.tool import read_json_file, save_json_file

default_system = conf_yaml['ui_conf']['default_system']
local_dict = conf_yaml['local_chat']['model_dict']
# -----------------video----------------------------
video_score_list = conf_yaml['video']['start_score']
breast_size_list = conf_yaml['video']['breast_size']
clothing_list = list(conf_yaml['video']['clothing'].values())
action_list = list(conf_yaml['video']['action'].values())
scene_list = list(conf_yaml['video']['scene'].values())
other_list = list(conf_yaml['video']['other'].values())
rag_list_config_path = conf_yaml['rag']['rag_list_config_path']
knowledge_base_info_save_path = conf_yaml['rag']['knowledge_base_info_save_path']

import os
os.environ["GRADIO_TEMP_DIR"] = os.path.join(os.getcwd(), "tmp")

def clear_session():
    return '', [], []

def refresh_rag():
    return list(read_rag_name_dict(rag_list_config_path).values())

def add_group_database(selected_documents_list, new_knowledge_base_name):
    # 所有知识库的记录，键为知识库名，值为构成该知识库的文章名列表
    all_knowledge_bases_record = {}
    if os.path.exists(knowledge_base_info_save_path):
        # 若文件存在，加载知识库信息
        all_knowledge_bases_record = read_json_file(knowledge_base_info_save_path)
    all_knowledge_bases_record_name_list = list(all_knowledge_bases_record.keys())
    if len(new_knowledge_base_name) == 0:
        gr.Warning('请输入新知识库的名字')
        return selected_documents_list, '', all_knowledge_bases_record, all_knowledge_bases_record_name_list
    else:
        # 将新建的知识库信息添加到记录中
        all_knowledge_bases_record[new_knowledge_base_name] = selected_documents_list
        # 保存更新后的知识库信息
        save_json_file(all_knowledge_bases_record, knowledge_base_info_save_path)
        gr.Info('新建知识库成功')
        return [], '', all_knowledge_bases_record, all_knowledge_bases_record_name_list

def delete_group_database(knowledge_bases_to_delete):
    # 所有知识库的记录，键为知识库名，值为构成该知识库的文章名列表
    if os.path.exists(knowledge_base_info_save_path):
        # 若文件存在，加载知识库信息
        all_knowledge_bases_record = read_json_file(knowledge_base_info_save_path)
    else:
        all_knowledge_bases_record = {}  # 这里修正为字典，因为后续操作是基于字典的键删除
    # 遍历要删除的知识库名称列表，从记录中删除对应的知识库
    for knowledge_base_name in knowledge_bases_to_delete:
        if knowledge_base_name in all_knowledge_bases_record:
            del all_knowledge_bases_record[knowledge_base_name]
    # 保存更新后的知识库信息
    save_json_file(all_knowledge_bases_record, knowledge_base_info_save_path)
    gr.Info('删除知识库成功')
    return [], all_knowledge_bases_record, list(all_knowledge_bases_record.keys())

with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.TabItem("聊天机器人"):
            with gr.TabItem("本地文字"):
                default_history = []
                history_state = gr.State(value=default_history)
                with gr.Row():
                    model_type = gr.Dropdown(list(local_dict.keys()), label="model type")
                    model_name = gr.Dropdown([], label="model name")
                    steam_check_box = gr.CheckboxGroup(["流式输出"], label="输出形式")

                @model_type.change(inputs=model_type, outputs=model_name)
                def update_cities(model_type):
                    model_list = list(local_dict[model_type])
                    return gr.Dropdown(choices=model_list, value=model_list[0], interactive=True)


                with gr.Row():
                    with gr.Column(scale=3):
                        system_input = gr.Textbox(value=default_system, lines=1, label='角色设置')

                chatbot = gr.Chatbot(label='qwen2', show_copy_button=True)
                with gr.Row():
                    with gr.Column(scale=1):
                        contextual_knowledge_list = list(read_json_file(knowledge_base_info_save_path).keys())
                        book_type = gr.Dropdown(choices=contextual_knowledge_list, label="上下文知识")
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
                rag_list_value_gradio = gr.JSON(visible=False)
                rag_list_value_gradio.value = read_rag_name_dict(rag_list_config_path)
                with gr.Row():
                    rag_checkboxgroup = gr.CheckboxGroup(choices=list(rag_list_value_gradio.value.values()), label="rag列表")

                with gr.Row():
                    with gr.Column(scale=1):
                        rag_delete_button = gr.Button(value='删除')
                with gr.Row():
                    with gr.Column(scale=1):
                        is_same_group = gr.Radio(["是", "否"], label="是否为同一个组", info='若选是，则多个文章视为一篇文章，否则将视为多篇文章')
                    with gr.Column(scale=4):
                        knowledge_name = gr.Textbox(lines=1, label='文章名字',
                                                    placeholder='给文章起个名字吧，不起的话，默认为上传的第一个文件的名字')
                rag_upload_file = gr.Files(label='上传文件，支持pdf,csv,md,jpg,docx格式，上传csv必须含title和content这两列。而且代码只解析这两列。')
                rag_submit_files_button = gr.Button(value='开始解析')
                rag_submit_files_button.click(
                    deal_mang_knowledge_files,
                    inputs=[rag_upload_file, is_same_group, knowledge_name],
                    outputs=[rag_list_value_gradio, is_same_group, knowledge_name, rag_upload_file]
                )

            with gr.TabItem('知识库'):
                knowledge_base_info_dict = {}
                if os.path.exists(knowledge_base_info_save_path):
                    # 若文件存在，加载知识库信息
                    knowledge_base_info_dict = read_json_file(knowledge_base_info_save_path)
                # 历史状态，保存已有的知识库名列表
                existing_knowledge_bases_state = gr.State(value=list(knowledge_base_info_dict.keys()))
                # 可选文章复选框组
                selectable_documents_checkbox_group = gr.CheckboxGroup(
                    choices=list(rag_list_value_gradio.value.values()), label='可选文章'
                )
                # 可选知识库复选框组
                selectable_knowledge_bases_checkbox_group = gr.CheckboxGroup(
                    choices=existing_knowledge_bases_state.value, label='已有知识库')
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
                             knowledge_base_info_json_table, existing_knowledge_bases_state]
                )
                # 删除知识库按钮点击事件
                delete_knowledge_base_button.click(
                    delete_group_database,
                    inputs=[selectable_knowledge_bases_checkbox_group],
                    outputs=[selectable_knowledge_bases_checkbox_group, knowledge_base_info_json_table,
                             existing_knowledge_bases_state]
                )
                rag_delete_button.click(
                    drop_lancedb_table,
                    inputs=rag_checkboxgroup,
                    outputs=[rag_list_value_gradio,knowledge_base_info_json_table]
                )

                @knowledge_base_info_json_table.change(inputs=knowledge_base_info_json_table,
                                                       outputs=[selectable_knowledge_bases_checkbox_group, book_type])
                def update_selectable_knowledge_bases(knowledge_base_info):
                    return gr.CheckboxGroup(choices=list(knowledge_base_info.keys()), label='可选知识库'),gr.Dropdown(choices=list(knowledge_base_info.keys()), label="上下文知识")

                @rag_list_value_gradio.change(inputs=rag_list_value_gradio,
                                                       outputs=[rag_checkboxgroup, selectable_documents_checkbox_group])
                def update_selectable_knowledge_bases(input_value):
                    return gr.CheckboxGroup(choices=list(input_value.keys()), label="rag管理"), gr.CheckboxGroup(choices=list(input_value.keys()), label='可选文章')

            with gr.TabItem('ToDo'):
                need_to_do_string = read_md_doc('./readme.md')
                gr.Markdown(need_to_do_string)

        with gr.TabItem("本地视频播放"):
            gr.Markdown("# Local Video Player")
            video_output = gr.Video(label="Play Local Video")
            video_path = gr.Textbox(visible=False)
            # 使用按钮触发加载本地视频文件
            load_button = gr.Button("Load Local Video")
            title_text = gr.Textbox(interactive=True, label='标题')
            start_radio = gr.Radio(video_score_list, label='评分')
            breast_radio = gr.Radio(breast_size_list, label='乳量')
            clothing_boxs = gr.CheckboxGroup(clothing_list, label="着装")
            action_boxs = gr.CheckboxGroup(action_list, label="动作")
            scene_boxs = gr.CheckboxGroup(scene_list, label="场景")
            other_boxs = gr.CheckboxGroup(other_list, label="其他")
            describe_text = gr.Textbox(interactive=True, label='备注')
            describe_button = gr.Button('提交')
            shutdown_button = gr.Button('关机')
            translation_title_text = gr.Textbox(interactive=True, label='翻译标题')
            with gr.Row():
                with gr.Column(scale=3):
                    with gr.Row():
                        start_cut_hour = gr.Textbox(label='s_hour', min_width=80, value='0')
                        start_cut_minute = gr.Textbox(label='s_min', min_width=80)
                        start_cut_second = gr.Textbox(label='s_sec', min_width=80)
                with gr.Column(scale=3):
                    with gr.Row():
                        end_cut_hour = gr.Textbox(label='e_hour', min_width=80, value='0')
                        end_cut_minute = gr.Textbox(label='e_min', min_width=80)
                        end_cut_second = gr.Textbox(label='e_sec', min_width=80)
                with gr.Column(scale=1):
                    with gr.Row():
                        with gr.Row():
                            video_mark_add_btn = gr.Button('mark', min_width=80)
                        with gr.Row():
                            video_mark_clean_btn = gr.ClearButton(components=[
                                start_cut_hour, start_cut_minute, start_cut_second,
                                end_cut_hour, end_cut_minute, end_cut_second
                            ])

            video_mark_add_btn.click(
                video_cut,
                inputs=[video_path, start_cut_hour, start_cut_minute, start_cut_second, end_cut_hour, end_cut_minute, end_cut_second],
                outputs=None
            )

            load_button.click(
                load_local_video,
                inputs=None,
                outputs=[
                    video_output, video_path,
                    start_radio, breast_radio,
                    clothing_boxs, action_boxs, scene_boxs, other_boxs,
                    describe_text, title_text, translation_title_text
                ])

            describe_button.click(
                mark_video_like,
                inputs=[
                    video_path,
                    start_radio, breast_radio,
                    clothing_boxs, action_boxs, scene_boxs, other_boxs,
                    describe_text],
                outputs=None)
            shutdown_button.click(
                shutdown_computer,
                inputs=None,
                outputs=None
            )

        with gr.TabItem('ocr check'):
            with gr.Row():
                ocr_model_type = gr.Dropdown(['StepfunOcr', 'RapidOCR'], label="选择模型")
            with gr.Row():
                img_input = gr.Image(type='filepath', label='Image')
                img_output = gr.Image()
                ocr_text = gr.Textbox(lines=20, max_lines=50, label='ocr_ouput', interactive=True, show_copy_button=True, container=True)
            img_input.upload(get_result_image, inputs=[img_input, ocr_model_type], outputs=[img_input, img_output, ocr_text])

demo.launch(server_name='0.0.0.0')