import gradio as gr
from local.rag.parse.voice_parse import standalone_voice_analysis

def voice_ui_show():
    with gr.Row():
        model_dir = gr.Textbox(value=r'C:\use\model\FireRedASR-AED-L', interactive=True)
    with gr.Row():
        speech_recognition_file = gr.File(label='上传一个mp3')
    with gr.Row():
        speech_recognition_output_text = gr.Textbox(label='语音转文字结果', show_copy_button=True)
    speech_recognition_file.upload(standalone_voice_analysis, inputs=[speech_recognition_file, model_dir],
                                   outputs=[speech_recognition_output_text, speech_recognition_file])
