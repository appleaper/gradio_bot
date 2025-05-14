import os
import re
import fitz
import hashlib
import pandas as pd
from tqdm import tqdm
from PIL import Image
from local.model.other_model.aigot import OCR_AiGot
from utils.tool import generate_unique_filename, hash_code, slice_string


ocr_class = OCR_AiGot()

def chunk_str(
        user_id,
        article_id,
        page_num,
        pdf_file_name,
        ocr_result,
        emb_model_name,
        database_type,
        embedding_class=None
):
    '''包装每一次编码的数据'''
    info = {}
    info['user_id'] = user_id
    info['article_id'] = article_id
    info['page_count'] = str(page_num)
    info['file_from'] = pdf_file_name
    info['title'] = ''
    info['content'] = ocr_result
    info['embed_model_name'] = emb_model_name
    info['platform'] = embedding_class.platform
    if database_type in ['milvus', 'lancedb']:
        # bge_m3是返回numpy，(1024,)  ollama返回的是[[...]]，形状是(1,1024)
        vector = embedding_class.parse_single_sentence(
            model_name=emb_model_name,
            sentence=ocr_result
        )[0]
        info['vector'] = vector
    else:
        info['vector'] = []
    info['database_type'] = database_type
    hash_content = user_id + article_id + ocr_result + emb_model_name + database_type
    info['hash_check'] = hashlib.sha256((hash_content).encode('utf-8')).hexdigest()
    return info

def parse_pdf_do(pdf_path, user_id, database_type, embedding_class, config_info):
    model_path = config_info['local_model_name_path_dict']['local_stepfun-aiGOT-OCR2_0']
    emb_model_name = embedding_class.model_name
    pdf_file_name = os.path.basename(pdf_path)
    article_id = hash_code(os.path.splitext(pdf_file_name)[0])

    ocr_class.init_mdoel(model_path)
    # 打开PDF文件
    pdf_document = fitz.open(pdf_path)
    # 遍历每一页

    info_list = []
    for page_num in tqdm(range(len(pdf_document)), total=len(pdf_document)):
        page = pdf_document.load_page(page_num)

        # 将页面转换为图片
        pix = page.get_pixmap()

        # 将pixmap转换为Pillow图像对象
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        output_path = generate_unique_filename('jpg')
        img.save(output_path)
        ocr_result = ocr_class.model.chat(ocr_class.tokenizer, output_path, ocr_type='format')
        if os.path.exists(output_path):
            # 删除文件
            os.remove(output_path)

        if len(ocr_result) > 0:
            text_list = slice_string(ocr_result, punctuation=r'[，。！？,.]')
            for text in text_list:
                info = chunk_str(user_id, article_id, page_num, pdf_file_name, text, emb_model_name, database_type, embedding_class)
                info_list.append(info)
    df = pd.DataFrame(info_list)
    return df

def llm_ocr(model_dir, file_path_list):
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    tmp_dir = os.path.join(project_dir, 'data', 'tmp')
    os.makedirs(tmp_dir, exist_ok=True)
    save_path = os.path.join(tmp_dir, 'result.csv')

    ocr_class.init_mdoel(model_dir)
    out_list = []
    ocr_str = ''

    for index, file_path in enumerate(file_path_list):
        info = {}
        ocr_result = ocr_class.parse_image(file_path)
        filename = os.path.basename(file_path)
        file_name, suffix = os.path.splitext(filename)
        info['file_name'] = file_name
        info['result'] = ocr_result
        out_list.append(info)
        if index == 0:
            ocr_str += ocr_result

    df = pd.DataFrame(out_list)
    df.to_csv(save_path, encoding='utf-8', index=False)
    ocr_class.unload_model()
    return ocr_str, save_path, file_path_list

if __name__ == '__main__':
    model_dir = r'C:\use\model\stepfun-aiGOT-OCR2_0'
    file_path_list = [r'C:\Users\APP\Desktop\gradio_bot\academic.jpg']
    ocr_str, save_path, file_path_list = llm_ocr(model_dir, file_path_list)
    print(ocr_str)