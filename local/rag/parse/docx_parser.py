import re
import os
import hashlib
import pandas as pd
from tqdm import tqdm
from docx import Document
from utils.tool import hash_code, slice_string, chunk_str

'''
pip install python-docx
在 Python 中方便地创建、读取和修改 Word 文档
'''
def split_docx_into_chunks(docx_path):
    doc = Document(docx_path)
    text = []

    # 遍历文档中的每个段落，并添加到文本列表中
    for para in doc.paragraphs:
        if len(para.text) == 0:
            continue
        else:
            text.append(para.text)
    return text

def parse_docx_do(file_path, user_id, database_type, emb_model_class):
    emb_model_name = emb_model_class.model_name

    file_name, _ = os.path.splitext(os.path.basename(file_path))
    article_id = hash_code(file_name)
    result_list = []
    sections = split_docx_into_chunks(file_path)
    for index, text in enumerate(sections):
        text_list = slice_string(text, punctuation=r'[，。！？,.]')
        for content in text_list:
            info = chunk_str('', content, user_id, article_id, '', emb_model_name, database_type, file_name,
                             emb_model_class)
            result_list.append(info)
    result_df = pd.DataFrame(result_list)
    return result_df

if __name__ == "__main__":
    # filename = "/home/pandas/snap/code/ragflow-new/sdk/python/test/test_sdk_api/test_data/test.docx"  # 替换为你的 Word 文档文件名
    # filename = '/media/pandas/DataHaven/code/doc/董秘办数据/大模型投关问答材料第一批240710/业绩发布及路演/公司2023年中期业绩发布会模拟问答V3.docx'
    filename = '/home/pandas/文档/大模型：不是万能钥匙，却是效率神器.docx'
    sections, tables = split_docx_into_chunks(filename)
    # sections返回的是列表，列表中每个元素是一个元祖，每个元祖有2个元素，第一个是文字，第二个是层级比如说heading，Normal
    print("Sections:", sections)