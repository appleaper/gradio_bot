import gradio as gr
from local.voice.speech_recognition_do import speech_recognition

def voice_ui_show():
    with gr.Row():
        speech_recognition_file = gr.File(label='上传一个mp3')
    with gr.Row():
        speech_recognition_output_text = gr.Textbox(label='语音转文字结果', show_copy_button=True)
    speech_recognition_file.upload(speech_recognition, inputs=speech_recognition_file,
                                   outputs=[speech_recognition_output_text, speech_recognition_file])
