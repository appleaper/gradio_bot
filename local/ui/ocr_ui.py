import gradio as gr
from local.rag.parse.image_parse import standalone_image_analysis

def ocr_ui_show():
    with gr.Row():
        model_dir = gr.Textbox(r'C:\use\model\stepfun-aiGOT-OCR2_0', interactive=True)
    with gr.Row():
        ocr_files = gr.Files(label='上传需要解析的图片,或者多张图片，或者PDF')
    with gr.Row():
        ocr_output_text = gr.Textbox(label='ocr检测结果，仅展示第一个文件结果', show_copy_button=True)
    with gr.Row():
        ocr_download_file = gr.File(label="点击下载示例文件")
    ocr_files.upload(standalone_image_analysis, inputs=[ocr_files, model_dir], outputs=[ocr_files, ocr_output_text, ocr_download_file])