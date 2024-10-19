import os.path
import gradio as gr
import pandas as pd
from local.local_api import local_chat
from video.video_play import load_local_video, mark_video_like
from video.shutdown_computer import shutdown_computer
from video.cut_video import video_cut
from ocr.ocr_model_select import get_result_image
from config import conf_yaml
from local.MiniCPM.minicpm_vl_detect.chat import minicpm_ui
from util.plot_data import create_pie_chart
from local.rag.pdf_rag import new_file_rag, drop_lancedb_table
from local.rag.image_group_rag import new_files_rag
from local.rag.util import read_rag_name_dict

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

def clear_session():
    return '', [], []

def refresh_rag():
    return list(read_rag_name_dict(rag_list_config_path).values())

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
                        contextual_knowledge_list = list(read_rag_name_dict(rag_list_config_path).values())
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
            # with gr.TabItem('书籍'):
            #     gr.HTML('<a href=file:///home/pandas/snap/code/RapidOcr/book/决策的艺术.html target="_blank">决策的艺术</a>')

            with gr.TabItem('rag'):
                with gr.Row():
                    rag_list_value = list(read_rag_name_dict(rag_list_config_path).values())
                    rag_checkboxgroup = gr.CheckboxGroup(choices=rag_list_value, label="rag列表")
                with gr.Row():
                    with gr.Column(scale=1):
                        rag_delete_button = gr.Button(value='删除')
                    rag_delete_button.click(
                        drop_lancedb_table,
                        inputs=rag_checkboxgroup,
                        outputs=[rag_checkboxgroup, book_type]
                    )

                with gr.Row():
                    rag_upload_file = gr.File(label='上传一个文件，支持pdf,csv,md格式')
                    rag_upload_file.upload(
                        new_file_rag,
                        inputs=rag_upload_file,
                        outputs=[rag_checkboxgroup, book_type]
                    )
                with gr.Row():
                    upload_files_group_name = gr.Textbox(label='组名', placeholder='给群组起一个名字')
                with gr.Row():
                    rag_upload_files = gr.Files(label='上传多张图片，支持png,jpg,jpeg格式')
                with gr.Row():
                    rag_submit_files_button = gr.Button(value='开始解析')
                rag_submit_files_button.click(
                    new_files_rag,
                    inputs=[rag_upload_files, upload_files_group_name],
                    outputs=[rag_checkboxgroup, book_type]
                )

            with gr.TabItem('ToDo'):
                need_to_do_string = '''
                1. 看一下JavaScript教程,
                2. 给聊天框加一个复制按钮  (完成)
                3. 想给长视频添加一些切分点，把长视频切分为不同的短视频   (完成)
                4. 获取影片标题之后，将标题翻译为中文    (完成)
                5. 给页面添加一个读取md的文本框，用示例来阅读   （完成失败）
                6. 输入pdf,csv就能挂载rag技术  (完成)
                '''
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

        with gr.TabItem('ocr check'):
            with gr.Row():
                ocr_model_type = gr.Dropdown(['StepfunOcr', 'RapidOCR'], label="选择模型")
            with gr.Row():
                img_input = gr.Image(type='filepath', label='Image')
                img_output = gr.Image()
                ocr_text = gr.Textbox(lines=20, max_lines=50, label='ocr_ouput', interactive=True, show_copy_button=True, container=True)
            img_input.upload(get_result_image, inputs=[img_input, ocr_model_type], outputs=[img_input, img_output, ocr_text])

demo.launch(server_name='0.0.0.0')