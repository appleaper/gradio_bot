import os
import re
import fitz
import uuid
import lancedb
import gradio as gr
import pandas as pd
from tqdm import tqdm
from PIL import Image
from docx import Document
from local.rag.rag_model import load_bge_model_cached, load_model_cached
from local.rag.util import save_rag_name_dict, read_rag_name_dict, write_rag_name_dict, save_rag_csv_name, save_info_to_lancedb, read_md_doc, split_by_heading

from config import conf_yaml
rag_config = conf_yaml['rag']
rag_top_k = rag_config['top_k']
rag_ocr_model_path = rag_config['ocr_model_path']
rag_data_csv_dir = rag_config['rag_data_csv_dir']
bge_model_path = rag_config['beg_model_path']
rag_database_name = rag_config['rag_database_name']
rag_list_config_path = rag_config['rag_list_config_path']


def generate_unique_filename(extension='jpg'):
    unique_filename = str(uuid.uuid4()) + '.' + extension
    return unique_filename

def pdf_to_images(pdf_path, progress=gr.Progress()):
    model_path = rag_ocr_model_path
    model, tokenizer = load_model_cached(model_path)
    # 打开PDF文件
    pdf_document = fitz.open(pdf_path)
    # 遍历每一页

    info_list = []
    progress(0, desc="Starting")
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
        progress(round((page_num+1) / len(pdf_document), 2))
    df = pd.DataFrame(info_list)
    data = read_rag_name_dict(rag_list_config_path)
    id = save_rag_name_dict(pdf_path, data, rag_list_config_path)
    df_save_path = save_rag_csv_name(df, rag_data_csv_dir, id, rag_list_config_path)
    return df, df_save_path, id

def dataframe_save_database(df, database_name, table_name):
    db = lancedb.connect(database_name)
    table_path = os.path.join(database_name, table_name + '.lance')
    if os.path.exists(table_path):
        db.drop_table(table_name)

    tb = db.create_table(table_name, data=df, exist_ok=True)
    row_count = tb.count_rows()
    if row_count>0:
        gr.Info(f'{row_count}条记录存入数据库')
    else:
        raise gr.Error('发生错误，没有成功将数据存入数据库')

def search_similar(database_name, table_name, test_str, top_k):
    model_bge = load_bge_model_cached(bge_model_path)
    vector = model_bge.encode(test_str, batch_size=1, max_length=8192)['dense_vecs'].tolist()
    db = lancedb.connect(database_name)
    tb = db.open_table(table_name)
    records = tb.search(vector).limit(top_k).to_pandas()
    return records

def csv_to_lancedb(csv_path, progress=gr.Progress()):
    try:
        df = pd.read_csv(csv_path, encoding='utf8')
    except:
        gr.Warning('文件读取失败')
        df = pd.DataFrame()
    csv_file_name = os.path.basename(csv_path)
    model_bge = load_bge_model_cached(bge_model_path)
    result_list = []
    for index, row in df.iterrows():
        info = {}
        input_str = f'标题/提问：{row.title}\n正文/回答:{row.content}\n'
        info['title'] = row['title']
        info['content'] = row['content']
        info['page_count'] = index+1
        info['vector'] = model_bge.encode(input_str, batch_size=1, max_length=8192)['dense_vecs'].tolist()
        info['file_from'] = csv_file_name
        result_list.append(info)
        progress(round((index + 1) / len(df), 2))
    result_df = pd.DataFrame(result_list)

    df_save_path = os.path.join(rag_data_csv_dir, csv_file_name + '.csv')
    result_df.to_csv(df_save_path, index=False, encoding='utf8')

    data = read_rag_name_dict(rag_list_config_path)
    id = save_rag_name_dict(csv_path, data, rag_list_config_path)
    return result_df, df_save_path, id

def save_data_to_lancedb(df, id):
    df = save_info_to_lancedb(df)
    dataframe_save_database(df, rag_database_name, id)
    data = read_rag_name_dict(rag_list_config_path)
    inverted_dict = {value: key for key, value in data.items()}
    return inverted_dict

def deal_markdown(md_path, progress=gr.Progress()):
    markdown_data = read_md_doc(md_path)
    markdown_data_list = split_by_heading(markdown_data, level=2)
    model_bge = load_bge_model_cached(bge_model_path)
    file_name = os.path.basename(md_path)
    csv_file_name = os.path.splitext(file_name)[0]
    result_list = []
    for index, markdown_line in tqdm(enumerate(markdown_data_list), total=len(markdown_data_list)):
        info = {}
        input_str = f'标题/提问：{markdown_line["title"]}\n正文/回答:{markdown_line["content"]}\n'
        info['title'] = markdown_line["title"]
        info['content'] = markdown_line["content"]
        info['page_count'] = ''
        info['vector'] = model_bge.encode(input_str, batch_size=1, max_length=8192)['dense_vecs'].tolist()
        info['file_from'] = file_name
        result_list.append(info)
        progress(round((index + 1) / len(markdown_data_list), 2))
    result_df = pd.DataFrame(result_list)
    df_save_path = os.path.join(rag_data_csv_dir, csv_file_name + '.csv')
    result_df.to_csv(df_save_path, index=False, encoding='utf8')
    data = read_rag_name_dict(rag_list_config_path)
    id = save_rag_name_dict(file_name, data, rag_list_config_path)
    return result_df, df_save_path, id

def split_docx_into_chunks(docx_path):
    """
    将.docx文件的内容分割成指定大小的块。

    参数:
    docx_path (str): .docx文件的路径。
    chunk_size (int): 每个块的大小（默认为8000个字符）。

    返回:
    list: 包含所有文本块的列表。
    """
    # 打开.docx文件
    doc = Document(docx_path)
    text = []

    # 遍历文档中的每个段落，并添加到文本列表中
    for para in doc.paragraphs:
        if len(para.text) == 0:
            continue
        else:
            text.append(para.text)
    return text

def deal_docx(doc_path, progress=gr.Progress()):
    model_bge = load_bge_model_cached(bge_model_path)
    file_name = os.path.basename(doc_path)
    docx_file_name = os.path.splitext(file_name)[0]
    result_list = []
    text_list = split_docx_into_chunks(doc_path)
    for index, text in tqdm(enumerate(text_list), total=len(text_list)):
        info = {}
        info['title'] = ''
        info['content'] = text
        info['page_count'] = ''
        info['vector'] = model_bge.encode(text, batch_size=1, max_length=8192)['dense_vecs'].tolist()
        info['file_from'] = file_name
        result_list.append(info)
        progress(round((index + 1) / len(text_list), 2))
    result_df = pd.DataFrame(result_list)
    df_save_path = os.path.join(rag_data_csv_dir, docx_file_name + '.csv')
    result_df.to_csv(df_save_path, index=False, encoding='utf8')
    data = read_rag_name_dict(rag_list_config_path)
    id = save_rag_name_dict(file_name, data, rag_list_config_path)
    return result_df, df_save_path, id

def new_file_rag(rag_file):
    upload_file, suffix = os.path.splitext(os.path.basename(rag_file))
    if suffix == '.pdf' or suffix=='.jpg':
        df, df_save_path, id = pdf_to_images(rag_file)
        inverted_dict = save_data_to_lancedb(df, id)
    elif suffix == '.csv':
        df, df_save_path, id = csv_to_lancedb(rag_file)
        inverted_dict = save_data_to_lancedb(df, id)
    elif suffix == '.md':
        df, df_save_path, id = deal_markdown(rag_file)
        inverted_dict = save_data_to_lancedb(df, id)
    elif suffix == '.docx':
        df, df_save_path, id = deal_docx(rag_file)
        inverted_dict = save_data_to_lancedb(df, id)
    else:
        raise gr.Error('上传的文件后缀格式不支持')
    rag_deal_list = list(inverted_dict.keys())
    return gr.CheckboxGroup(choices=rag_deal_list, label="rag列表"), gr.Dropdown(choices=rag_deal_list, label="上下文知识")

def drop_lancedb_table(table_name_list):
    db = lancedb.connect(rag_database_name)
    data = read_rag_name_dict(rag_list_config_path)
    inverted_dict = {value: key for key, value in data.items()}
    for table_name in table_name_list:
        if table_name in inverted_dict:
            table_path = os.path.join(rag_database_name, inverted_dict[table_name] + '.lance')
            if os.path.exists(table_path):
                db.drop_table(inverted_dict[table_name])
                del data[inverted_dict[table_name]]
                del inverted_dict[table_name]
                gr.Info(f'成功删除{table_name}')

            csv_drop_path = os.path.join(rag_data_csv_dir, table_name + '.csv')
            if os.path.exists(csv_drop_path):
                os.remove(csv_drop_path)
    write_rag_name_dict(rag_list_config_path, data)
    rag_deal_list = list(inverted_dict.keys())
    return gr.CheckboxGroup(choices=rag_deal_list, label="rag列表"), gr.Dropdown(choices=rag_deal_list, label="上下文知识")

def select_lancedb_table():
    data = read_rag_name_dict(rag_list_config_path)
    inverted_dict = {value: key for key, value in data.items()}
    return gr.Dropdown(choices=list(inverted_dict.keys()), label="上下文知识")

if __name__ == '__main__':
    pass