import os
import fitz
import uuid
import hashlib
import pandas as pd
from tqdm import tqdm
from PIL import Image
from local.rag.rag_model import load_bge_model_cached, load_model_cached

from utils.config_init import rag_ocr_model_path, bge_model_path



def generate_unique_filename(extension='jpg'):
    unique_filename = str(uuid.uuid4()) + '.' + extension
    return unique_filename

def parse_pdf_do(pdf_path, id, user_id):
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
        info['user_id'] = user_id
        info['article_id'] = id
        info['page_count'] = str(page_num)
        info['file_from'] = pdf_path
        info['title'] = ''
        info['content'] = ocr_result
        info['vector'] = model_bge.encode(ocr_result, batch_size=1, max_length=8192)['dense_vecs'].tolist()
        info['hash_check'] = hashlib.sha256((user_id+id+ocr_result).encode('utf-8')).hexdigest()
        info_list.append(info)
    df = pd.DataFrame(info_list)
    return df