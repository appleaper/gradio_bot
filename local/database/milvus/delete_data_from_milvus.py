import copy
from utils.tool import encrypt_username,reverse_dict,read_json_file, save_json_file
from local.database.milvus.milvus_article_management import MilvusArticleManager

def drop_milvus_table(need_detele_articles, all_articles_dict, user_name):
    manager = MilvusArticleManager()
    user_id = encrypt_username(user_name)
    need_delete_articles_id_list = []
    reverse_all_articles_dict = reverse_dict(all_articles_dict)
    all_articles_dict_copy = copy.deepcopy(all_articles_dict)
    for article in need_detele_articles:
        if article in all_articles_dict.values():
            article_id = reverse_all_articles_dict[article]
            need_delete_articles_id_list.append(article_id)
            del all_articles_dict_copy[article_id]
    manager.delete_data_by_article_id(user_id, need_delete_articles_id_list)

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