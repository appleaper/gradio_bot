import os
import fitz
import uuid
import pandas as pd
import gradio as gr
from tqdm import tqdm
from PIL import Image
from config import conf_yaml
from local.rag.rag_model import load_bge_model_cached, load_model_cached
from local.rag.util import save_rag_name_dict, save_rag_csv_name, read_rag_name_dict


rag_config = conf_yaml['rag']
rag_ocr_model_path = rag_config['ocr_model_path']
bge_model_path = rag_config['beg_model_path']
rag_list_config_path = rag_config['rag_list_config_path']
rag_data_csv_dir = rag_config['rag_data_csv_dir']

def generate_unique_filename(extension='jpg'):
    unique_filename = str(uuid.uuid4()) + '.' + extension
    return unique_filename

def parse_pdf_do(pdf_path):
    model_path = rag_ocr_model_path
    model, tokenizer = load_model_cached(model_path)
    # 打开PDF文件
    pdf_document = fitz.open(pdf_path)
    # 遍历每一页

    info_list = []
    model_bge = load_bge_model_cached(bge_model_path)
    for page_num in tqdm(range(len(pdf_document)), total=len(pdf_document)):
        page = pdf_document.load_page(page_num)

        # 将页面转换为图片
        pix = page.get_pixmap()

        # 将pixmap转换为Pillow图像对象
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        output_path = generate_unique_filename('jpg')
        img.save(output_path)
        ocr_result = model.chat(tokenizer, output_path, ocr_type='format')
        if os.path.exists(output_path):
            # 删除文件
            os.remove(output_path)
        info = {}
        info['page_count'] = page_num
        info['file_from'] = pdf_path
        info['title'] = ''
        info['content'] = ocr_result
        info['vector'] = model_bge.encode(ocr_result, batch_size=1, max_length=8192)['dense_vecs'].tolist()
        info_list.append(info)
    df = pd.DataFrame(info_list)
    return df