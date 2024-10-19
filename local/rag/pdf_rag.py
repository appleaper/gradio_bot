import os
import fitz
import uuid
import hashlib
import lancedb
import functools
import numpy as np
import gradio as gr
import pandas as pd
from tqdm import tqdm
from PIL import Image
from config import conf_yaml
from FlagEmbedding import BGEM3FlagModel
from transformers import AutoModel, AutoTokenizer
from local.qwen.qwen_api import qwen_model_init,qwen_model_detect
from util.tool import save_rag_name_dict, read_rag_name_dict, write_rag_name_dict, save_rag_csv_name, save_rag_group_name, save_rag_group_csv_name

rag_top_k = 3
rag_ocr_model_path = '/home/pandas/snap/model/stepfun-aiGOT-OCR2_0'
rag_data_csv_dir = '/home/pandas/snap/code/RapidOcr/database_data/rag/data_csv'
bge_model_path = '/home/pandas/snap/model/BAAIbge-m3'
rag_database_name = '/home/pandas/snap/code/RapidOcr/database_data/rag/database'
rag_list_config_path = '/home/pandas/snap/code/RapidOcr/config/rag_list.json'

def cached_model_loader(func):
    cache = {}
    @functools.wraps(func)
    def wrapper(model_path):
        # 计算输入的哈希值
        input_hash = hashlib.md5(model_path.encode()).hexdigest()
        if input_hash not in cache:
            # 如果输入的哈希值不在缓存中，则加载模型并缓存结果
            cache[input_hash] = func(model_path)
        return cache[input_hash]
    return wrapper
@cached_model_loader
def load_model_cached(model_path):
    return load_model(model_path)

@cached_model_loader
def load_bge_model_cached(model_path):
    model_bge = BGEM3FlagModel(model_path, use_fp16=True)
    return model_bge

def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, low_cpu_mem_usage=True, device_map='cuda',
                                      use_safetensors=True, pad_token_id=tokenizer.eos_token_id)
    model = model.eval().cuda()
    return model, tokenizer

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

def save_info_to_lancedb(df):
    result_list = []
    for index, row in tqdm(df.iterrows(), total=len(df)):
        info = {}
        info['page_count'] = row.page_count
        info['file_from'] = row.file_from
        info['title'] = row.title
        info['content'] = row.content
        info['vector'] = np.array(row.vector)
        result_list.append(info)
    df_out = pd.DataFrame(result_list)
    return df_out

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
        gr.Error('发生错误，没有成功将数据存入数据库')

def search_similar(database_name, table_name, test_str, top_k):
    model_bge = load_bge_model_cached(bge_model_path)
    vector = model_bge.encode(test_str, batch_size=1, max_length=8192)['dense_vecs'].tolist()
    db = lancedb.connect(database_name)
    tb = db.open_table(table_name)
    records = tb.search(vector).limit(top_k).to_pandas()
    return records

def chat_with_model(test_str, rag_records):
    qwen_model_path = '/home/pandas/snap/model/qwen/Qwen2-7B-Instruct-AWQ'

    rag_str = ''
    for record_index, record_row in rag_records.iterrows():
        rag_str += record_row['content']

    model, tokenizer = qwen_model_init(qwen_model_path)
    messages = [
        {"role": "system", "content": '你是一个有用的助手'},
        {'role': 'user', 'content': rag_str + '以上为参考信息，回答以下用户提问的问题,回答时尽可能的简洁，并根据参考信息' + test_str}
    ]
    response = qwen_model_detect(messages, model, tokenizer)

def csv_to_lancedb(csv_path, model_path, progress=gr.Progress()):
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
def new_file_rag(rag_file):
    upload_file, suffix = os.path.splitext(os.path.basename(rag_file))
    if suffix == '.pdf' or suffix=='.jpg':
        df, df_save_path, id = pdf_to_images(rag_file)
        inverted_dict = save_data_to_lancedb(df, id)
    elif suffix == '.csv':
        df, df_save_path, id = csv_to_lancedb(rag_file, bge_model_path)
        inverted_dict = save_data_to_lancedb(df, id)
    elif suffix == '.md':
        pass
    else:
        raise gr.Error('上传的文件后缀格式不支持')
    rag_deal_list = list(inverted_dict.keys())
    return gr.CheckboxGroup(choices=rag_deal_list, label="rag列表"), gr.Dropdown(choices=rag_deal_list, label="上下文知识")

def deal_images_group(group_name, rag_files, progress=gr.Progress()):

    info_list = []
    model, tokenizer = load_model_cached(rag_ocr_model_path)
    model_bge = load_bge_model_cached(bge_model_path)
    for index, file_name in tqdm(enumerate(rag_files), total=len(rag_files)):
        upload_file, suffix = os.path.splitext(os.path.basename(file_name))
        if suffix in ['.jpg', 'jpeg', 'png']:
            ocr_result = model.chat(tokenizer, file_name, ocr_type='format')
            info = {}
            info['title'] = ''
            info['content'] = ocr_result
            info['page_count'] = ''
            info['vector'] = model_bge.encode(ocr_result, batch_size=1, max_length=8192)['dense_vecs'].tolist()
            info['file_from'] = upload_file + suffix
            info_list.append(info)
        progress(round((index + 1) / len(rag_files), 2))
    df = pd.DataFrame(info_list)
    data = read_rag_name_dict(rag_list_config_path)
    id = save_rag_group_name(group_name, data, rag_list_config_path)
    df_save_path = save_rag_group_csv_name(df, rag_data_csv_dir, id, rag_list_config_path)
    return df, df_save_path, id

def create_or_add_data_to_lancedb(rag_database_name, table_name, df):
    db = lancedb.connect(rag_database_name)
    table_path = os.path.join(rag_database_name, table_name + '.lance')
    if not os.path.exists(table_path):
        tb = db.create_table(table_name, data=df, exist_ok=True)
        row_count = tb.count_rows()
        gr.Info(f'创建新的数据表，并写入{row_count}条数据')
    else:
        tb = db.open_table(table_name)
        old_count = tb.count_rows()
        tb.add(data=df)
        now_count = tb.count_rows()
        gr.Info(f'写入{now_count - old_count}条记录, 现有{now_count}条记录')

def get_rag_now_list():
    data = read_rag_name_dict(rag_list_config_path)
    inverted_dict = {value: key for key, value in data.items()}
    rag_deal_list = list(inverted_dict.keys())
    return rag_deal_list

def new_files_rag(rag_files, group_name):
    if group_name == '':
        gr.Warning('请给群组起一个名字')
        rag_deal_list = get_rag_now_list()
        return gr.CheckboxGroup(choices=rag_deal_list, label="rag列表"), gr.Dropdown(choices=rag_deal_list, label="上下文知识")
    else:
        df, df_save_path, id = deal_images_group(group_name, rag_files)
        df = save_info_to_lancedb(df)
        create_or_add_data_to_lancedb(rag_database_name, id, df)
        rag_deal_list = get_rag_now_list()
        return gr.CheckboxGroup(choices=rag_deal_list, label="rag列表"), gr.Dropdown(choices=rag_deal_list,
                                                                                     label="上下文知识")

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
    # data = read_rag_name_dict(rag_list_config_path)
    # save_rag_name_dict('规则书.pdf', data, rag_list_config_path)
    # data = read_rag_name_dict(rag_list_config_path)
    # print(data)
    print(1)

    # pdf_path = '/home/pandas/下载/规则书.pdf'
    # pdf_to_images(pdf_path)

    # save_dir = '/home/pandas/snap/code/RapidOcr/database_data/rag/data_csv'
    # df = pd.read_csv('/home/pandas/snap/code/RapidOcr/database_data/rag/data_csv/规则书.csv')
    # save_info_to_lancedb(df, save_dir)

    # df = pd.read_csv('/home/pandas/snap/code/RapidOcr/database_data/rag/data_csv/规则书_emb.csv')
    # database_name = '/home/pandas/snap/code/RapidOcr/database_data/rag/database'
    # table_name = 'gui_ze_shu'
    # dataframe_save_database(df, database_name, table_name)


    # table_name = 'gui_ze_shu'
    # test_str = '每轮空投有什么物资？'
    #
    # model_path = '/home/pandas/snap/model/BAAIbge-m3'
    # rag_records = search_similar(database_name, table_name, model_path, test_str, top_k)
    #
    # chat_with_model(test_str, rag_records)