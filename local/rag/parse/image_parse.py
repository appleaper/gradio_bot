import os
import pandas as pd
from local.model.other_model.aigot import OCR_AiGot
from utils.tool import hash_code, slice_string, chunk_str

ocr_class = OCR_AiGot()

def parse_image_do(file_path, user_id, database_type, embedding_class, config_info):
    model_path = config_info['local_model_name_path_dict']['local_stepfun-aiGOT-OCR2_0']
    emb_model_name = embedding_class.model_name
    file_name, _ = os.path.splitext(os.path.basename(file_path))
    article_id = hash_code(file_name)
    ocr_class.init_mdoel(model_path)
    # 打开PDF文件
    ocr_result = ocr_class.model.chat(ocr_class.tokenizer, file_path, ocr_type='format')
    info_list = []
    if len(ocr_result) > 0:
        text_list = slice_string(ocr_result, punctuation=r'[，。！？,.]')
        for text in text_list:
            info = chunk_str('', text, user_id, article_id, '', emb_model_name, database_type, file_name, embedding_class)
            info_list.append(info)
    df = pd.DataFrame(info_list)
    return df

def standalone_image_analysis(file_path_list, model_dir):
    '''独立的解析图片'''
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    tmp_dir = os.path.join(project_dir, 'data', 'tmp')
    save_path = os.path.join(tmp_dir, 'result.csv')
    ocr_class.init_mdoel(model_dir)
    out_list = []
    ocr_str = ''
    for index, file_path in enumerate(file_path_list):

        info = {}
        ocr_result = ocr_class.model.chat(ocr_class.tokenizer, file_path, ocr_type='format')
        filename = os.path.basename(file_path)
        file_name, suffix = os.path.splitext(filename)
        info['file_name'] = file_name
        info['result'] = ocr_result
        out_list.append(info)
        if index == 0:
            ocr_str += ocr_result
    df = pd.DataFrame(out_list)
    df.to_csv(save_path, encoding='utf8', index=False)
    ocr_class.unload_model()
    return [], ocr_str, save_path