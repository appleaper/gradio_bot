import lancedb
import numpy as np
import gradio as gr
import pandas as pd
from local.database.mysql.mysql_article_management import MySQLDatabase
from local.embedding_model.embedding_init import load_rag_model
from utils.tool import encrypt_username,read_user_info_dict, reverse_dict
from local.database.milvus.milvus_article_management import MilvusArticleManager
from local.database.es.es_article_management import ElasticsearchManager
from utils.config_init import akb_conf_class



def get_info_from_vector(textbox, book_type, rag_model, database_name, top_k, user_name, database_type):
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
    all_knowledge_bases_record = read_user_info_dict(user_name, akb_conf_class.kb_article_map_path)
    inverted_dict = reverse_dict(read_user_info_dict(user_name, akb_conf_class.articles_user_path))
    vector = rag_model.encode(textbox, batch_size=1, max_length=8192)['dense_vecs']
    akb_conf_class.get_database_config(database_type)
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
        article_range = all_knowledge_bases_record[book_type]
        articles_id_list = []
        for article_name in article_range:
            articles_id_list.append(inverted_dict[article_name])
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

def get_info_from_mysql(search_content, search_range, top_k, user_name):
    '''根据条件从mysql中获取信息'''
    top_k = int(top_k)
    all_knowledge_bases_record = read_user_info_dict(user_name, akb_conf_class.kb_article_map_path)
    inverted_dict = reverse_dict(read_user_info_dict(user_name, akb_conf_class.articles_user_path))
    mysql_database = MySQLDatabase(
        host=akb_conf_class.mysql_host,
        user=akb_conf_class.mysql_user,
        password=akb_conf_class.mysql_password,
        port=akb_conf_class.mysql_port
    )
    article_name_list = all_knowledge_bases_record[search_range]
    article_id_list = []
    user_id = encrypt_username(user_name)
    for article_name in article_name_list:
        article_id_list.append(inverted_dict[article_name])
    mysql_database.connect()
    mysql_database.use_database(akb_conf_class.mysql_database_name)
    result = mysql_database.select_data(akb_conf_class.mysql_article_table_info_name, user_id, search_content, article_id_list, top_k)
    result_list = []
    for row in result:
        info = {}
        info['title'] = row[2]
        info['content'] = row[3]
        info['file_from'] = row[5]
        result_list.append(info)
    return pd.DataFrame(result_list)

def get_info_from_es(search_content, search_range, top_k, user_name):
    '''根据条件从es中获取信息'''
    top_k = int(top_k)
    all_knowledge_bases_record = read_user_info_dict(user_name, akb_conf_class.kb_article_map_path)
    inverted_dict = reverse_dict(read_user_info_dict(user_name, akb_conf_class.articles_user_path))
    article_name_list = all_knowledge_bases_record[search_range]
    article_id_list = []
    user_id = encrypt_username(user_name)
    for article_name in article_name_list:
        article_id_list.append(inverted_dict[article_name])

    es_class = ElasticsearchManager(akb_conf_class.es_index_name)
    result = es_class.query_data(user_id, article_id_list, search_content, top_k)
    df = pd.DataFrame(result)
    df = df[['title', 'content', 'file_from']]
    return df

def search_data_from_database_do(database_type, search_content, search_range, search_tok_k, request: gr.Request):
    user_name = request.username
    if database_type=='lancedb':
        rag_model, rag_tokenizer = load_rag_model('bge_m3')
        akb_conf_class.get_database_config('lancedb')
        rag_df = get_info_from_vector(search_content, search_range, rag_model, akb_conf_class.database_dir, search_tok_k, user_name, 'lancedb')
    elif database_type == 'milvus':
        rag_model, rag_tokenizer = load_rag_model('bge_m3')
        akb_conf_class.get_database_config('milvus')
        rag_df = get_info_from_vector(search_content, search_range, rag_model, akb_conf_class.database_dir, search_tok_k, user_name,
                                      'milvus')
    elif database_type =='mysql':
        rag_df = get_info_from_mysql(search_content, search_range, search_tok_k, user_name)
    elif database_type == 'es':
        rag_df = get_info_from_es(search_content, search_range, search_tok_k, user_name)
    else:
        raise gr.Error('暂时不支持')
    return rag_df


def get_user_select_info(evt: gr.SelectData):
    '''获取用户点击的行的信息'''
    title = evt.row_value[0]
    content = evt.row_value[1]
    file_from = evt.row_value[2]
    return title, content, file_from
