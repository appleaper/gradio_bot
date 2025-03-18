
import gradio as gr
from config import conf_yaml
from video.video_play import load_local_video, mark_video_like
from video.shutdown_computer import shutdown_computer
from video.cut_video import video_cut
# -----------------video----------------------------
video_score_list = conf_yaml['video']['start_score']
breast_size_list = conf_yaml['video']['breast_size']
clothing_list = list(conf_yaml['video']['clothing'].values())
action_list = list(conf_yaml['video']['action'].values())
scene_list = list(conf_yaml['video']['scene'].values())
other_list = list(conf_yaml['video']['other'].values())

def adult_ui_show():
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
        gr.Markdown(value='提供视频切分，s开始表示切割的开始时间，e表示切割结尾')
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
                    video_mark_add_btn = gr.Button('开始切割视频', min_width=80)
                with gr.Row():
                    video_mark_clean_btn = gr.ClearButton(components=[
                        start_cut_hour, start_cut_minute, start_cut_second,
                        end_cut_hour, end_cut_minute, end_cut_second
                    ], value='清空')

    video_mark_add_btn.click(
        video_cut,
        inputs=[video_path, start_cut_hour, start_cut_minute, start_cut_second, end_cut_hour, end_cut_minute,
                end_cut_second],
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