import re
import os
import hashlib
import pandas as pd
from tqdm import tqdm
from io import BytesIO
from docx import Document
from collections import Counter
from local.rag.rag_model import load_bge_model_cached


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

def split_text_into_chunks(text_list):
    combined_text = '\n'.join(text_list)
    chunks = []
    for i in range(0, len(combined_text), 1000):
        chunks.append(combined_text[i:i + 1000])
    return chunks

def parse_docx_do(doc_path, id, user_id, database_type):
    if database_type in ['lancedb', 'milvus']:
        model_bge = load_bge_model_cached(bge_m3_model_path)
    file_name = os.path.basename(doc_path)
    result_list = []
    sections = split_docx_into_chunks(doc_path)
    chunks = split_text_into_chunks(sections)
    for index, text in tqdm(enumerate(chunks), total=len(chunks)):
        if len(text) == 0:
            continue
        info = {}
        info['user_id'] = user_id
        info['article_id'] = id
        info['title'] = ''
        info['content'] = text
        info['page_count'] = ''
        if database_type in ['lancedb', 'milvus']:
            info['vector'] = model_bge.encode(text, batch_size=1, max_length=8192)['dense_vecs'].tolist()
        else:
            info['vector'] = []
        info['file_from'] = file_name
        info['hash_check'] = hashlib.sha256((user_id+id+text).encode('utf-8')).hexdigest()
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