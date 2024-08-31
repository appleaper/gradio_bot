import gradio as gr
from chat_api.qwen.qwen_response import qwen_chat
from chat_api.baidu.baidu_api import baidu_chat
from local.local_api import local_chat
from video.video_play import load_local_video, mark_video_like
from ocr.ocr_show import show_result
from config import conf_yaml

default_system = conf_yaml['ui_conf']['default_system']
qwen_dict = conf_yaml['qwen_api_chat']
qianfan_dict = conf_yaml['baidu_api_chat']
local_dict = conf_yaml['local_chat']['model_dict']
rag_list = conf_yaml['rag']['rag_info_name']

def clear_session():
    return '', [], []

def room_set(model_dict, model_chat):
    default_history = []
    history_state = gr.State(value=default_history)
    # gr.Markdown("""<center><font size=8>文字聊天</center>""")
    with gr.Row():
        model_type = gr.Dropdown(list(model_dict.keys()), label="model type")
        model_name = gr.Dropdown([], label="model name")
        steam_check_box = gr.CheckboxGroup(["流式输出"], label="输出形式")

    @model_type.change(inputs=model_type, outputs=model_name)
    def update_cities(model_type):
        model_list = list(model_dict[model_type])
        return gr.Dropdown(choices=model_list, value=model_list[0], interactive=True)

    with gr.Row():
        with gr.Column(scale=3):
            system_input = gr.Textbox(value=default_system, lines=1, label='角色设置')
        with gr.Column(scale=1):
            modify_system = gr.Button("🛠️ Set system prompt and clear history", scale=2)
        system_state = gr.Textbox(value=default_system, visible=False)

    chatbot = gr.Chatbot(label='qwen2')
    with gr.Row():
        with gr.Column(scale=1):
            book_type = gr.Dropdown(rag_list, label="上下文知识")
        with gr.Column(scale=4):
            textbox = gr.Textbox(lines=1, label='输入')

    with gr.Row():
        clear_history = gr.Button("🧹 清除历史")
        sumbit = gr.Button("🚀 发送")

    textbox.submit(model_chat,
                   inputs=[textbox, chatbot, system_input, history_state, model_type, model_name, steam_check_box, book_type],
                   outputs=[textbox, chatbot, history_state])

    sumbit.click(model_chat,
                 inputs=[textbox, chatbot, system_input, history_state, model_type, model_name, steam_check_box, book_type],
                 outputs=[textbox, chatbot, history_state])

    clear_history.click(fn=clear_session,
                        inputs=[],
                        outputs=[textbox, chatbot, history_state])

with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.TabItem("聊天机器人"):
            with gr.TabItem("本地"):
                room_set(local_dict, local_chat)
            with gr.TabItem("阿里"):
                room_set(qwen_dict, qwen_chat)
            with gr.TabItem("百度"):
                room_set(qianfan_dict, baidu_chat)

        with gr.TabItem("本地视频播放"):
            gr.Markdown("# Local Video Player")
            video_output = gr.Video(label="Play Local Video")
            video_path = gr.Textbox(visible=False)
            # 使用按钮触发加载本地视频文件
            load_button = gr.Button("Load Local Video")

            start_radio = gr.Radio(["1分", "2分", "3分", "4分", '5分'], label='评分')
            type_check_boxs = gr.CheckboxGroup(["高根", '丝袜','女仆装','学生装','ol','巨乳','中乳','小乳','多人','情趣'], label="")
            describe_text = gr.Textbox(interactive=True)
            describe_button = gr.Button('提交')
            load_button.click(load_local_video, inputs=None, outputs=[video_output, video_path, start_radio, type_check_boxs, describe_text])
            describe_button.click(mark_video_like, inputs=[start_radio, type_check_boxs, describe_text, video_path], outputs=None)

        with gr.TabItem('ocr check'):
            with gr.Row():
                img_input = gr.Image(type='filepath', label='Image')
                img_output = gr.Image()
                ocr_text = gr.Textbox(lines=20, max_lines=50, label='ocr_ouput', interactive=True, show_copy_button=True, container=True)
            img_input.upload(show_result, inputs=img_input, outputs=[img_input, img_output, ocr_text])



demo.launch(server_name='0.0.0.0')