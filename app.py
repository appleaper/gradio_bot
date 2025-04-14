
import gradio as gr

from local.ui.ocr_ui import ocr_ui_show
from local.ui.config_ui import config_ui_show
from local.ui.voice_ui import voice_ui_show
from local.ui.chat_ui import chat_ui_show

with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.TabItem("聊天机器人"):
            chat_ui_show(demo)

        # with gr.TabItem("本地视频播放"):
        #     adult_ui_show()

        with gr.TabItem('ocr识别'):
            ocr_ui_show()

        with gr.TabItem('语音识别'):
            voice_ui_show()

        with gr.TabItem('默认配置'):
            config_ui_show(demo)


# demo.launch(server_name='0.0.0.0')
demo.launch(
    server_name='0.0.0.0',
    auth=[('a', "a"), ('b', 'b'), ('pandas', '123')],
    server_port=7680,
    allowed_paths=[r'D:\video']
)