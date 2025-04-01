import copy

import gradio as gr

from utils.tool import encrypt_username,reverse_dict,read_json_file, save_json_file
from local.database.mysql.mysql_article_management import init_mysql

def drop_mysql_table(need_detele_articles, all_articles_dict, user_name):
    manager = init_mysql()
    user_id = encrypt_username(user_name)
    need_delete_articles_id_list = []
    reverse_all_articles_dict = reverse_dict(all_articles_dict)
    all_articles_dict_copy = copy.deepcopy(all_articles_dict)
    for article in need_detele_articles:
        if article in all_articles_dict.values():
            article_id = reverse_all_articles_dict[article]
            need_delete_articles_id_list.append(article_id)
            del all_articles_dict_copy[article_id]
    res_flag = manager.delete_data_by_user_and_article_ids(akb_conf_class.mysql_article_table_info_name,user_id, need_delete_articles_id_list)
    if res_flag:
        gr.Info(f'{"、".join(need_delete_articles_id_list)} delete successful!')
    else:
        gr.Warning(f'{"、".join(need_delete_articles_id_list)} delete failed!')
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
    articles_user_mapping_dict[user_name] = all_articles_dict_copy
    save_json_file(articles_user_mapping_dict, akb_conf_class.articles_user_path)
    return all_articles_dict_copy, knowledge_json