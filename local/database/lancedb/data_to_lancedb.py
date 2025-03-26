import os
import copy
import lancedb
import gradio as gr
import pandas as pd
from utils.tool import read_json_file, save_json_file, reverse_dict
from utils.config_init import akb_conf_class, rag_data_csv_dir

def create_or_add_data_to_lancedb(rag_database_name, table_name, df):
    '''创建数据表，并插入数据，若数据存在就跳过'''
    db = lancedb.connect(rag_database_name)
    table_path = os.path.join(rag_database_name, table_name + '.lance')
    if not os.path.exists(table_path):
        tb = db.create_table(table_name, data=df, exist_ok=True)
        row_count = tb.count_rows()
        gr.Info(f'创建新的数据表，并写入{row_count}条数据')
        return df
    else:
        tb = db.open_table(table_name)
        old_count = tb.count_rows()
        need_to_add_data_list = []
        for index, row in df.iterrows():
            hash_check = row.hash_check
            result = tb.search().where(f"hash_check='{hash_check}'").to_list()
            if len(result) > 0:
                continue
            else:
                need_to_add_data_list.append(row)
        if len(need_to_add_data_list)!=0:
            add_df = pd.DataFrame(need_to_add_data_list)
            tb.add(data=add_df)
            now_count = tb.count_rows()
            gr.Info(f'写入{now_count - old_count}条记录, 现有{now_count}条记录')
            return add_df
        else:
            return pd.DataFrame([])

def drop_lancedb_table(need_detele_articles, all_articles_dict, user_name):
    '''
    删除文章
    '''
    db = lancedb.connect(akb_conf_class.database_dir)
    reverse_all_articles_dict = reverse_dict(all_articles_dict)
    for table_name in need_detele_articles:
        if table_name in reverse_all_articles_dict.keys():
            table_path = os.path.join(akb_conf_class.database_dir, reverse_all_articles_dict[table_name] + '.lance')
            if os.path.exists(table_path):
                db.drop_table(reverse_all_articles_dict[table_name])
                del reverse_all_articles_dict[table_name]
                gr.Info(f'成功删除{table_name}')
            else:
                del reverse_all_articles_dict[table_name]
                gr.Warning(f'数据不存在，但依旧执行删除')
            csv_drop_path = os.path.join(rag_data_csv_dir, table_name + '.csv')
            if os.path.exists(csv_drop_path):
                os.remove(csv_drop_path)

    knowledge_json = read_json_file(akb_conf_class.kb_article_map_path)
    for knowledge_name in list(knowledge_json[user_name].keys()):
        articles_list = knowledge_json[user_name][knowledge_name]
        for need_detele_article in need_detele_articles:
            if need_detele_article in articles_list:
                articles_list.remove(need_detele_article)
        if len(articles_list)==0:
            # 如果列表为空，删除该键值对
            del knowledge_json[user_name][knowledge_name]

    save_json_file(knowledge_json, akb_conf_class.kb_article_map_path)

    articles_user_mapping_dict = read_json_file(akb_conf_class.articles_user_path)
    articles_user_mapping_dict[user_name] = reverse_dict(reverse_all_articles_dict)
    save_json_file(articles_user_mapping_dict, akb_conf_class.articles_user_path)
    return reverse_dict(reverse_all_articles_dict), knowledge_json

if __name__ == '__main__':
    from utils.tool import load_data
    info = load_data('/home/pandas/snap/code/RapidOcr/local/database/lancedb/temp.pkl')
    rag_database_name, table_name, df = info['rag_database_name'], info['table_name'], info['df']
    create_or_add_data_to_lancedb(rag_database_name, table_name, df.iloc[20:30])