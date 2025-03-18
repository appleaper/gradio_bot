import re
import os
import hashlib
import pandas as pd
from tqdm import tqdm
from io import BytesIO
from docx import Document
from collections import Counter
from local.rag.rag_model import load_bge_model_cached

from utils.config_init import bge_model_path

'''
pip install python-docx
在 Python 中方便地创建、读取和修改 Word 文档
'''


def parse_docx_do(doc_path, id, user_id):
    model_bge = load_bge_model_cached(bge_model_path)
    file_name = os.path.basename(doc_path)
    result_list = []
    sections_list = []
    parser = RAGFlowDocxParser()
    sections, tables = parser(doc_path)
    temp_str = ''
    for text, style in sections:
        if len(temp_str) < 1000:
            temp_str += text + '\n'
        else:
            sections_list.append(temp_str)
            temp_str = text + '\n'
    sections_list.extend(tables)
    for index, text in tqdm(enumerate(sections_list), total=len(sections)):
        if len(text) == 0:
            continue
        info = {}
        info['user_id'] = user_id
        info['article_id'] = id
        info['title'] = ''
        info['content'] = text
        info['page_count'] = ''
        info['vector'] = model_bge.encode(text, batch_size=1, max_length=8192)['dense_vecs'].tolist()
        info['file_from'] = file_name
        info['hash_check'] = hashlib.sha256((user_id+id+text).encode('utf-8')).hexdigest()
        result_list.append(info)

    result_df = pd.DataFrame(result_list)

    return result_df

if __name__ == "__main__":
    parser = RAGFlowDocxParser()
    # filename = "/home/pandas/snap/code/ragflow-new/sdk/python/test/test_sdk_api/test_data/test.docx"  # 替换为你的 Word 文档文件名
    # filename = '/media/pandas/DataHaven/code/doc/董秘办数据/大模型投关问答材料第一批240710/业绩发布及路演/公司2023年中期业绩发布会模拟问答V3.docx'
    filename = '/home/pandas/文档/大模型：不是万能钥匙，却是效率神器.docx'
    sections, tables = parser(filename)
    # sections返回的是列表，列表中每个元素是一个元祖，每个元祖有2个元素，第一个是文字，第二个是层级比如说heading，Normal
    print("Sections:", sections)
    print("Tables:", tables)