import os.path
import gradio as gr
from local.rag.parse.image_parse import standalone_image_analysis
from local.function.doclayout_yolo.analysis_of_plate_surface import psa_analysis
from local.function.paddle_ocr_torch.predict_cls import imgs_cls_predict
from local.function.paddle_ocr_torch.predict_det import imgs_det_predict
from local.function.paddle_ocr_torch.predict_rec import imgs_rec_predict
from local.function.paddle_ocr_torch.predict_system import imgs_sys_predict
from utils.tool import read_yaml

from client.llm_ocr_client import analyze_images_client, psa_analysis_client, imgs_cls_predict_client, text_detection_client, text_recognition_client


config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'function', 'paddle_ocr_torch', 'model.yaml')
conf = read_yaml(config_path)
det_model_list = list(conf['det'].keys())
rec_model_list = list(conf['rec'].keys())
sys_model_list = list(conf['sys'].keys())

def img_show(img_path):
    return img_path

def ocr_ui_show():
    with gr.TabItem('大模型ocr识别'):
        with gr.Group():
            with gr.Row():
                model_dir = gr.Textbox(r'C:\use\model\stepfun-aiGOT-OCR2_0', interactive=True)
            with gr.Row():
                ocr_files = gr.Files(label='上传需要解析的图片,或者多张图片，或者PDF')
            with gr.Row():
                ocr_image = gr.Image()
            with gr.Row():
                ocr_output_text = gr.Textbox(label='ocr检测结果，仅展示第一个文件结果', show_copy_button=True)
            with gr.Row():
                ocr_download_file = gr.File(label="点击下载示例文件")
            ocr_files.upload(
                analyze_images_client,
                inputs=[ocr_files, model_dir],
                outputs=[ocr_files, ocr_output_text, ocr_download_file, ocr_image]
            )

    with gr.TabItem('板面分析'):
        with gr.Row():
            psa_img_path = gr.File(label='上传图片')
        with gr.Row():
            psa_model_path = gr.Textbox(
                value=r'C:\use\model\DocLayout-YOLO-DocStructBench\doclayout_yolo_docstructbench_imgsz1024.pt',
                interactive=True,
                label='模型位置'
            )
        with gr.Row():
            psa_analysis_button = gr.Button(value='开始分析')
        with gr.Row():
            psa_img_show = gr.Image()
        psa_analysis_button.click(
            fn=psa_analysis_client,
            inputs=[psa_img_path, psa_model_path],
            outputs=psa_img_show
        )

    with gr.TabItem('方向检测'):
        with gr.Group():
            with gr.Row():
                cls_model_type = gr.Textbox(value=r'C:\use\model\ocr\pt_models\ch_ptocr_mobile_v2.0_cls_infer.pth', interactive=True, label='模型路径')
            with gr.Row():
                cls_img_path = gr.File(label='上传图片,他只能检测是0度还是180度，图片名不含中文，而且有可能读取失败')
                cls_img_show = gr.Image()
            with gr.Row():
                cls_button = gr.Button(value='开始检测')
            with gr.Row():
                cls_result = gr.Textbox(label='检测结果')

        cls_button.click(
            imgs_cls_predict_client,
            inputs=[cls_img_path, cls_model_type],
            outputs=[cls_result]
        )
        cls_img_path.upload(
            img_show,
            inputs=cls_img_path,
            outputs=cls_img_show
        )

    with gr.TabItem('文字检测'):
        with gr.Group():
            with gr.Row():
                det_model_type = gr.Dropdown(choices=det_model_list, label='模型名称', interactive=True, filterable=True)
            with gr.Row():
                det_img_path = gr.File(label='上传图片, 路径不含中文')
                det_img_show = gr.Image()
            with gr.Row():
                det_button = gr.Button(value='开始检测')
            with gr.Row():
                det_result = gr.Image()


        det_button.click(
            text_detection_client,
            inputs=[det_img_path, det_model_type],
            outputs=[det_result]
        )
        det_img_path.change(
            fn=img_show,
            inputs=det_img_path,
            outputs=det_img_show
        )

    with gr.TabItem('文字识别'):
        with gr.Group():
            with gr.Row():
                rec_model_type = gr.Dropdown(choices=rec_model_list, label='模型名称', interactive=True, filterable=True)
            with gr.Row():
                rec_img_path = gr.File(label='上传图片，路径不含中文, 只能识别几个字的那种很小的图片')
                rec_img_show = gr.Image()
            with gr.Row():
                rec_button = gr.Button(value='开始检测')
            with gr.Row():
                rec_text_result = gr.Textbox()

        rec_button.click(
            fn=text_recognition_client,
            inputs=[rec_img_path, rec_model_type],
            outputs=[rec_text_result]
        )
        rec_img_path.upload(
            fn=img_show,
            inputs=rec_img_path,
            outputs=rec_img_show
        )

    with gr.TabItem('小模型ocr识别'):
        with gr.Group():
            with gr.Row():
                sys_model_type = gr.Dropdown(choices=sys_model_list, label='方式组合', interactive=True, filterable=True)
            with gr.Row():
                sys_img_path = gr.File(label='上传图片，路径不含中文')
                sys_img_show = gr.Image()
            with gr.Row():
                sys_button = gr.Button(value='开始检测')
            with gr.Row():
                sys_img_result = gr.Image()
            with gr.Row():
                sys_df_result = gr.DataFrame()



        sys_button.click(
            fn=imgs_sys_predict,
            inputs=[sys_img_path, sys_model_type],
            outputs=[sys_df_result, sys_img_result]
        )
        sys_img_path.upload(
            fn=img_show,
            inputs=sys_img_path,
            outputs=sys_img_show
        )

if __name__ == '__main__':
    with gr.Blocks() as demo:
        ocr_ui_show()
    demo.launch(allowed_paths=[
        r'C:\use\code\RapidOcr_small\data\tmp',
        r'C:\use\code\PaddleOCR2Pytorch-main\doc',
        r'C:\use\code\RapidOcr_small\local\ui'
    ])