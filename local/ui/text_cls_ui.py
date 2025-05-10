
import os.path
import gradio as gr
import pandas as pd
from PIL import Image

from server.function.text_clssification.build_data import train,predict
from client.text_cls_client import text_cls_train_client

def test(path):
    df = pd.read_csv(path, encoding='utf-8')
    return df.iloc[:100]

def show_data_detail(save_dir):
    img1_path = os.path.join(save_dir, 'content_length_histogram.png')
    img2_path = os.path.join(save_dir, 'cls_counts_pie_chart.png')
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)
    return img1, img2

def text_cls_show():
    with gr.Group():
        with gr.Row():
            model_dir = gr.Textbox(r'C:\use\model\bert_ch', interactive=True, label='预训练模型位置')
        with gr.Row():
            data_path = gr.File(label='上传数据集，格式为.csv，只含2列，一列是content，一列是cls，只接受编码格式为utf-8的')
        with gr.Row():
            batch_size = gr.Number(value=4, interactive=True, label='一次训练的批次大小')
            num_epochs = gr.Number(value=30, interactive=True, label='训练的轮数')
        with gr.Row():
            save_dir = gr.Textbox(r'C:\Users\APP\Desktop\gradio_bot\output', interactive=True, label='模型保存路径')
        with gr.Row():
            train_button = gr.Button(value='开始训练')
        with gr.Row():
            content_len_img = gr.Image(visible=False)
            cls_img = gr.Image(visible=False)
        with gr.Row():
            train_info = gr.DataFrame()

        with gr.Row():
            show_df = gr.Button(value='展示数据')
        with gr.Row():
            data_show = gr.DataFrame(column_widths='50%')
    with gr.Group():
        with gr.Row():
            pre_input_text = gr.Textbox(value='', interactive=True, label='输入要预测的内容文字')
        with gr.Row():
            pre_button = gr.Button(value='开始预测')
        with gr.Row():
            pre_result = gr.JSON(label='分类结果,置信度范围[0-1]')


    train_button.click(
        text_cls_train_client,
        inputs=[data_path, model_dir, batch_size, num_epochs, save_dir],
        outputs=[train_info]
    ).then(
        show_data_detail,
        inputs=[save_dir],
        outputs = [content_len_img, cls_img]
    )

    show_df.click(
        test,
        inputs=[data_path],
        outputs=[data_show]
    )

    pre_button.click(
        predict,
        inputs=[model_dir, save_dir, pre_input_text],
        outputs=[pre_result]
    )

if __name__ == '__main__':
    '''
    我现在的逻辑是配制有改动就保存到本地中。
    '''
    with gr.Blocks() as demo:
        text_cls_show()
    demo.launch(
        # auth=[('a', "a"), ('b', 'b'), ('pandas', '123')],
    )