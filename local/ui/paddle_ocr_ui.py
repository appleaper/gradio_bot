import os.path

import gradio as gr
from local.function.paddle_ocr_torch.predict_cls import imgs_cls_predict
from local.function.paddle_ocr_torch.predict_det import imgs_det_predict
from local.function.paddle_ocr_torch.predict_rec import imgs_rec_predict
from local.function.paddle_ocr_torch.predict_system import imgs_sys_predict
from utils.tool import read_yaml

config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'function', 'paddle_ocr_torch', 'model.yaml')
conf = read_yaml(config_path)
det_model_list = list(conf['det'].keys())
rec_model_list = list(conf['rec'].keys())
sys_model_list = list(conf['sys'].keys())

def img_show(img_path):
    return img_path

def paddle_ocr_show():
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
            imgs_cls_predict,
            inputs=[cls_img_path, cls_model_type],
            outputs=[cls_result, cls_img_path]
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
            imgs_det_predict,
            inputs=[det_img_path, det_model_type],
            outputs=[det_result, det_img_path]
        )
        det_img_path.upload(
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
            fn=imgs_rec_predict,
            inputs=[rec_img_path, rec_model_type],
            outputs=[rec_text_result, rec_img_path]
        )
        rec_img_path.upload(
            fn=img_show,
            inputs=rec_img_path,
            outputs=rec_img_show
        )

    with gr.TabItem('ocr识别'):
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
            outputs=[sys_df_result, sys_img_result, sys_img_path]
        )
        sys_img_path.upload(
            fn=img_show,
            inputs=sys_img_path,
            outputs=sys_img_show
        )

if __name__ == '__main__':
    '''
    我现在的逻辑是配制有改动就保存到本地中。
    '''
    with gr.Blocks() as demo:
        paddle_ocr_show()
    demo.launch(allowed_paths=[r'C:\use\code\RapidOcr_small\data\tmp'])