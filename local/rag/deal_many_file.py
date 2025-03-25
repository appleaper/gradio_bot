import os
import gradio as gr
from tqdm import tqdm
import pandas as pd
from local.rag.parse.pdf_parse import parse_pdf_do
from local.rag.parse.csv_parse import parse_csv_do
from local.rag.parse.markdown_parse import parse_markdown_do
from local.rag.parse.docx_parser import parse_docx_do
from local.rag.parse.image_parse import parse_image_do
from local.rag.parse.voice_parse import parse_voice_do
from local.rag.parse.video_parse import parse_video_do
from utils.tool import encrypt_username, save_rag_group_name, reverse_dict, read_user_info_dict, save_json_file, read_json_file
from local.database.lancedb.data_to_lancedb import create_or_add_data_to_lancedb, drop_lancedb_table
from local.database.milvus.milvus_article_management import MilvusArticleManager
from local.database.milvus.delete_data_from_milvus import drop_milvus_table
from utils.config_init import rag_data_csv_dir, database_dir, articles_user_path, kb_article_map_path, database_type


def save_rag_group_csv_name(df2, rag_data_csv_dir, id, rag_list_config_path, user_name):
    history_rag_dict = read_user_info_dict(user_name, rag_list_config_path)
    csv_name = history_rag_dict[id]
    save_path = os.path.join(rag_data_csv_dir, user_name, csv_name + '.csv')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if os.path.exists(save_path):
        df1 = pd.read_csv(save_path, encoding='utf8')
        df2 = pd.concat([df1, df2], ignore_index=True)
        df2 = df2.drop_duplicates(subset=['hash_check'])
        df2.to_csv(save_path, index=False, encoding='utf8')
    else:
        df2.to_csv(save_path, index=False, encoding='utf8')
    return save_path



def deal_mang_knowledge_files(rag_upload_files, is_same_group, knowledge_name, request: gr.Request, progress=gr.Progress()):
    '''将各种格式的东西转为embbing并存入数据库中'''
    user_name = request.username
    user_id = encrypt_username(user_name)

    if knowledge_name == '':
        knowledge_name = os.path.basename(rag_upload_files[0])
        article_name, file_suffix = os.path.splitext(knowledge_name)
    else:
        article_name = knowledge_name
    articles_user_mapping_dict = read_json_file(articles_user_path)
    id = save_rag_group_name(user_name, article_name, articles_user_mapping_dict, articles_user_path)
    if database_type=='milvus':
        manager = MilvusArticleManager()
    for file_index, file_name in tqdm(enumerate(rag_upload_files), total=len(rag_upload_files)):
        upload_file, suffix = os.path.splitext(os.path.basename(file_name))
        if suffix == '.pdf':
            df = parse_pdf_do(file_name, id, user_id)
        elif suffix in ['.csv', '.xlsx']:
            df = parse_csv_do(file_name, id, user_id)
        elif suffix == '.md':
            df = parse_markdown_do(file_name, id, user_id)
        elif suffix == '.docx':
            df = parse_docx_do(file_name, id, user_id)
        elif suffix in ['.jpg', '.jpeg', '.png']:
            df = parse_image_do(file_name, id, user_id)
        elif suffix == '.mp3':
            df = parse_voice_do(file_name, id, user_id)
        elif suffix == '.mp4':
            df = parse_video_do(file_name, id, user_id)
        else:
            gr.Warning(f'{os.path.basename(file_name)}不支持解析')
            continue
        if is_same_group == '否' and file_index!=0:
            data = read_user_info_dict(user_name, articles_user_path)
            article_name, article_suffix = os.path.splitext(os.path.basename(file_name))
            id = save_rag_group_name(user_name, article_name, data, articles_user_path)

        if database_type == 'lancedb':
            save_df = create_or_add_data_to_lancedb(database_dir, id, df)
        else:
            manager.create_collection(user_id)
            save_df = manager.insert_data_to_milvus(df, user_id)
        save_df['database_type'] = database_type
        save_rag_group_csv_name(save_df, rag_data_csv_dir, id, articles_user_path, user_name)
        progress(round((file_index + 1) / len(rag_upload_files), 2))
    return articles_user_mapping_dict[user_name], None, None, []


def delete_article_from_database(need_detele_articles, all_articles_dict, request: gr.Request):
    user_name = request.username
    if database_type == 'lancedb':
        all_articles_dict, articles_user_mapping_dict = drop_lancedb_table(need_detele_articles, all_articles_dict, user_name)
    else:
        all_articles_dict, articles_user_mapping_dict = drop_milvus_table(need_detele_articles, all_articles_dict, user_name)
    return all_articles_dict, articles_user_mapping_dict[user_name]


def add_group_database(selected_documents_list, new_knowledge_base_name, request: gr.Request):
    # 所有知识库的记录，键为知识库名，值为构成该知识库的文章名列表
    user_name = request.username
    if os.path.exists(kb_article_map_path):
        # 若文件存在，加载知识库信息
        all_knowledge_bases_record_with_user = read_json_file(kb_article_map_path)
    else:
        raise gr.Error(f'{kb_article_map_path} not exist!')
    user_kb = all_knowledge_bases_record_with_user[user_name]
    if len(new_knowledge_base_name) == 0:
        gr.Warning('请输入新知识库的名字')
        return selected_documents_list, '', user_kb, user_kb
    else:
        # 将新建的知识库信息添加到记录中
        all_knowledge_bases_record_with_user[user_name][new_knowledge_base_name] = selected_documents_list
        # 保存更新后的知识库信息
        save_json_file(all_knowledge_bases_record_with_user, kb_article_map_path)
        gr.Info('新建知识库成功')
        return [], '', all_knowledge_bases_record_with_user[user_name], all_knowledge_bases_record_with_user[user_name]

def delete_group_database(knowledge_bases_to_delete, request: gr.Request):
    # 所有知识库的记录，键为知识库名，值为构成该知识库的文章名列表
    user_name = request.username
    if os.path.exists(kb_article_map_path):
        # 若文件存在，加载知识库信息
        all_knowledge_bases_record = read_json_file(kb_article_map_path)
    else:
        raise gr.Error(f'{kb_article_map_path} not exist')
    # 遍历要删除的知识库名称列表，从记录中删除对应的知识库
    for knowledge_base_name in knowledge_bases_to_delete:
        if knowledge_base_name in all_knowledge_bases_record[user_name]:
            del all_knowledge_bases_record[user_name][knowledge_base_name]
    # 保存更新后的知识库信息
    save_json_file(all_knowledge_bases_record, kb_article_map_path)
    gr.Info('删除知识库成功')
    return [], all_knowledge_bases_record[user_name], all_knowledge_bases_record[user_name]