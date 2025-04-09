import gradio as gr
from local.ocr.ocr_model_select import get_result_image

def ocr_ui_show():
    with gr.Row():
        ocr_files = gr.Files(label='上传需要解析的图片,或者多张图片，或者PDF')
    with gr.Row():
        ocr_output_text = gr.Textbox(label='ocr检测结果，仅展示第一个文件结果', show_copy_button=True)
    with gr.Row():
        ocr_download_file = gr.File(label="点击下载示例文件")
    ocr_files.upload(get_result_image, inputs=ocr_files, outputs=[ocr_output_text, ocr_download_file])