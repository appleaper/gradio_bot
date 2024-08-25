import gradio as gr
from chat_api.qwen.qwen_response import qwen_chat
from chat_api.baidu.baidu_api import baidu_chat
from local.local_api import local_chat
from video.video_play import load_local_video

default_system = 'You are a helpful assistant.'
qwen_dict = {
    "qwen2": ["1.5b-instruct", "0.5b-instruct"],
    "qwen1.5": ["1.8b-chat", "0.5b-chat"],
}

qianfan_dict = {
    'ernie-speed': ['ernie_speed', 'ernie-speed-128k'],
    'ernie-lite': ['eb-instant', 'ernie-lite-8k'],
    'ernie-tiny':['ernie-tiny-8k'],
    'other':['yi_34b_chat']
}

local_dict = {
    'qwen':['1.5B', '7B'],
    'llama3':['8b']
}

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
    textbox = gr.Textbox(lines=1, label='输入')

    with gr.Row():
        clear_history = gr.Button("🧹 清除历史")
        sumbit = gr.Button("🚀 发送")

    textbox.submit(model_chat,
                   inputs=[textbox, chatbot, system_input, history_state, model_type, model_name, steam_check_box],
                   outputs=[textbox, chatbot, history_state])

    sumbit.click(model_chat,
                 inputs=[textbox, chatbot, system_input, history_state, model_type, model_name, steam_check_box],
                 outputs=[textbox, chatbot, history_state])

    clear_history.click(fn=clear_session,
                        inputs=[],
                        outputs=[textbox, chatbot, history_state])

with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.TabItem("本地"):
            room_set(local_dict, local_chat)

        with gr.TabItem("本地视频播放"):
            gr.Markdown("# Local Video Player")
            video_output = gr.Video(label="Play Local Video")
            # 使用按钮触发加载本地视频文件
            load_button = gr.Button("Load Local Video")
            load_button.click(load_local_video, inputs=None, outputs=video_output)

        with gr.TabItem("阿里"):
            room_set(qwen_dict, qwen_chat)

        with gr.TabItem("百度"):
            room_set(qianfan_dict, baidu_chat)

demo.launch(server_name='0.0.0.0')