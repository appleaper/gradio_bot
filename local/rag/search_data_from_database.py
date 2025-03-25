import lancedb
import numpy as np
import gradio as gr
import pandas as pd
from local.database.mysql.mysql_article_management import MySQLDatabase
from utils.config_init import database_type
from local.embedding_model.embedding_init import load_rag_model
from utils.tool import encrypt_username,read_user_info_dict, reverse_dict
from local.database.milvus.milvus_article_management import MilvusArticleManager

from local.chat_model.chat_do import add_rag_info, database_dir, kb_article_map_path, articles_user_path

def get_info_from_vector(textbox, book_type, rag_model, database_name, top_k, user_name):
    '''
    暂时去掉了模型审核内容是否相关的部分
    :param textbox: 用户提问
    :param book_type: 知识库
    :param rag_model: rag向量模型
    :param database_name: lancedb数据库存放的位置
    :param top_k: 取多少条相关记录
    :return: rag文字
    '''
    top_k = int(top_k)
    # model, tokenizer = load_model_cached('qwen2.5-0.5B-Instruct')
    all_knowledge_bases_record = read_user_info_dict(user_name, kb_article_map_path)
    inverted_dict = reverse_dict(read_user_info_dict(user_name, articles_user_path))
    vector = rag_model.encode(textbox, batch_size=1, max_length=8192)['dense_vecs']

    if database_type=='lancedb':
        db = lancedb.connect(database_name)
        records_df = pd.DataFrame()
        for table_name in all_knowledge_bases_record[book_type]:
            tb = db.open_table(inverted_dict[table_name])
            records = tb.search(vector).limit(top_k).to_pandas()
            records_df = pd.concat((records, records_df))
        sorted_records_df = records_df.sort_values(by='_distance').iloc[:top_k]
    elif database_type == 'milvus':
        user_id = encrypt_username(user_name)
        manager = MilvusArticleManager()
        articles_id_list = list(inverted_dict.values())
        vector = np.array(vector, dtype=np.float32)
        res = manager.search_vectors_with_articles(user_id, vector, articles_id_list, limit=top_k)
        sorted_records_df = pd.DataFrame(res)
    else:
        raise gr.Error(f'{database_type} not support!')
    result_list = []
    for _, row in sorted_records_df.iterrows():
        info = {}
        info['title'] = row['title']
        info['content'] = row['content']
        info['file_from'] = row['file_from']
        result_list.append(info)
    return pd.DataFrame(result_list)



def search_data_from_database_do(search_type, search_content, search_range, search_tok_k, request: gr.Request):
    user_name = request.username
    if search_type=='向量搜索':
        rag_model, rag_tokenizer = load_rag_model('bge_m3')
        rag_df = get_info_from_vector(search_content, search_range, rag_model, database_dir, search_tok_k, user_name)
        return rag_df
    elif search_type =='mysql搜索':
        mysql_database = MySQLDatabase()
    else:
        raise gr.Error('暂时不支持')


def get_user_select_info(evt: gr.SelectData):
    '''获取用户点击的行的信息'''
    title = evt.row_value[0]
    content = evt.row_value[1]
    file_from = evt.row_value[2]
    return title, content, file_from
