import os
import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch
from utils.tool import singleton

@singleton
class ElasticsearchManager:
    def __init__(self, index_name, host='localhost', port=9200, scheme='http'):
        self.es = Elasticsearch([{'host': host, 'port': port, 'scheme': scheme}])
        self.index_name = index_name
        if not self.es.indices.exists(index=self.index_name):
            self.es.indices.create(index=self.index_name)

    def insert_data(self, article_id, user_id, title, content, page_count, file_from, hash_check):
        if self.es.exists(index=self.index_name, id=hash_check):
            print(f"数据已存在，hash_check: {hash_check}，不插入。")
            return 0
        data = {
            "article_id": article_id,
            "user_id": user_id,
            "title": title,
            "content": content,
            "page_count": page_count,
            "file_from": file_from,
            "hash_check": hash_check
        }
        try:
            self.es.index(index=self.index_name, id=hash_check, body=data)
            return 1
        except Exception as e:
            print(f"数据插入失败，错误信息: {e}")
            return 0

    def insert_data_format_df(self, df):
        save_df = []
        for index, row in df.iterrows():
            article_id = row['article_id']
            user_id = row['user_id']
            title = self._replace_nan(row['title'])
            content = self._replace_nan(row['content'])
            page_count = self._replace_nan(row['page_count'])
            file_from = self._replace_nan(row['file_from'])
            hash_check = row['hash_check']

            res_flag = self.insert_data(
                article_id=article_id,
                user_id=user_id,
                title=title,
                content=content,
                page_count=page_count,
                file_from=file_from,
                hash_check=hash_check
            )
            if res_flag:
                save_df.append(row)
        return pd.DataFrame(save_df)

    def query_data(self, user_id, article_id, query_content, size=10):
        if not isinstance(article_id, list):
            article_id = [article_id]
        query = {
            "bool": {
                "must": [
                    {"term": {"user_id": user_id}},
                    {"terms": {"article_id": article_id}},
                    {
                        "multi_match": {
                            "query": query_content,
                            "fields": ["title", "content"]
                        }
                    }
                ]
            }
        }
        # print(f"查询条件: {query}")
        try:
            result = self.es.search(index=self.index_name, body={"query": query}, size=size)
            hits = result['hits']['hits']
            return [hit['_source'] for hit in hits]
        except Exception as e:
            print(f"查询失败，错误信息: {e}")
            return []


    def delete_data(self, user_id, article_id):
        if not isinstance(article_id, list):
            article_id = [article_id]
        query = {
            "bool": {
                "must": [
                    {"term": {"user_id": user_id}},
                    {"terms": {"article_id": article_id}}
                ]
            }
        }
        print(f"删除条件: {query}")
        try:
            result = self.es.delete_by_query(index=self.index_name, body={"query": query})
            print(f"删除成功，删除数量: {result['deleted']}")
            return True
        except Exception as e:
            print(f"删除失败，错误信息: {e}")
            return False

    def _replace_nan(self, value):
        if isinstance(value, float) and np.isnan(value):
            return None
        return value

    def count_data(self):
        try:
            result = self.es.count(index=self.index_name)
            return result['count']
        except Exception as e:
            print(f"统计数据总量失败，错误信息: {e}")
            return 0

    def delete_all_data(self):
        query = {
            "match_all": {}
        }
        print(f"删除所有数据的条件: {query}")
        try:
            result = self.es.delete_by_query(index=self.index_name, body={"query": query})
            print(f"删除所有数据成功，删除数量: {result['deleted']}")
        except Exception as e:
            print(f"删除所有数据失败，错误信息: {e}")


def insert_data(user_name):
    es_manager = ElasticsearchManager(index_name="article_index")
    doc_csv_dir = r'C:\use\code\RapidOcr_small\data\rag\data_csv\pandas'
    akb_conf_class.get_database_config('es')
    akb_conf_class.init_article_user_and_kb_mapping_file()
    for file in tqdm(os.listdir(doc_csv_dir)):
        data = read_json_file(akb_conf_class.articles_user_path)
        article_name, article_suffix = os.path.splitext(os.path.basename(file))
        article_id = save_rag_group_name(user_name, article_name, data, akb_conf_class.articles_user_path)

        file_path = os.path.join(doc_csv_dir, file)
        df = pd.read_csv(file_path, encoding='utf8')
        for index, row in df.iterrows():
            article_id = es_manager._replace_nan(article_id)
            user_id = es_manager._replace_nan(row['user_id'])
            title = es_manager._replace_nan(row['title'])
            content = es_manager._replace_nan(row['content'])
            page_count = es_manager._replace_nan(row['page_count'])
            file_from = es_manager._replace_nan(row['file_from'])
            hash_check = es_manager._replace_nan(row['hash_check'])
            es_manager.insert_data(
                article_id=article_id,
                user_id=user_id,
                title=title,
                content=content,
                page_count=page_count,
                file_from=file_from,
                hash_check=hash_check
            )

if __name__ == "__main__":
    from tqdm import tqdm
    from utils.tool import read_json_file, save_rag_group_name, encrypt_username
    from utils.config_init import akb_conf_class
    user_name = 'pandas'
    user_id = encrypt_username(user_name)
    insert_data(user_name)

    # # 查询数据示例
    # es_manager = ElasticsearchManager(index_name="article_index")
    # es_manager.delete_all_data()
    # count = es_manager.count_data()
    # print(f"Elasticsearch 中 {es_manager.index_name} 索引的数据总量为: {count}")

    # results = es_manager.query_data(
    #     user_id=user_id,
    #     article_id=['6f417500','e40590c0','9e0beecb'],
    #     query_content="武则天",
    #     size=10
    # )
    # print("查询结果:", results)

    # # 删除数据示例
    # es_manager.delete_data(user_id=user_id, article_id=['6f417500','e40590c0','9e0beecb'])
    # count = es_manager.count_data()
    # print(f"Elasticsearch 中 {es_manager.index_name} 索引的数据总量为: {count}")