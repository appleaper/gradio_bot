import os
import gradio as gr
import pandas as pd
import numpy as np
from pymilvus import MilvusClient, DataType
from utils.tool import singleton

@singleton
class MilvusArticleManager:
    def __init__(self, host='127.0.0.1', port='19530'):
        self.host = host
        self.port = port
        self.client = self.connect_milvus(host, port)

    def connect_milvus(self, host, port):
        '''连接milvus'''
        try:
            # 建立与 Milvus 的连接，连接别名为 "default"
            client = MilvusClient(
                uri=f"http://{host}:{port}",
                token="root:Milvus"
            )
            return client
        except Exception as e:
            # 若连接失败，抛出 Gradio 错误信息
            raise gr.Error('milvus can not connect')

    def create_collection(self, collection_name):
        '''创建集合'''

        schema = MilvusClient.create_schema(
            auto_id=False,  # id禁止自动增加
            enable_dynamic_field=False,  # 禁止自动字段
        )
        schema.add_field(field_name="article_id", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="user_id", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=256)
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=20000)
        schema.add_field(field_name="page_count", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name="file_from", datatype=DataType.VARCHAR, max_length=256)
        schema.add_field(field_name="hash_check", datatype=DataType.VARCHAR, max_length=64, is_primary=True,
                         auto_id=False)

        # 准备索引参数
        index_params = self.client.prepare_index_params()

        # 为字符串字段添加索引，使用 STL_SORT 索引类型
        string_fields = ["article_id", "user_id", "title", "page_count", "file_from", "hash_check"]
        for field in string_fields:
            index_params.add_index(
                field_name=field,  # Name of the scalar field to be indexed
                index_type="",  # Type of index to be created. For auto indexing, leave it empty or omit this parameter.
            )

        # 为向量字段添加索引，使用 IVF_FLAT 索引类型
        index_params.add_index(
            field_name="vector",
            metric_type="COSINE",
            index_type="IVF_FLAT",
            index_name="vector_index",
            params={"nlist": 128}
        )

        if self.client.has_collection(collection_name=collection_name):
            pass
        else:
            # 创建集合（类似数据库），集合名字为collection_name，表的初始化也放进去了
            self.client.create_collection(
                collection_name=collection_name,
                schema=schema,
            )
            gr.Info(f'create {collection_name} successful!')

            self.client.create_index(
                collection_name=collection_name,
                index_params=index_params,
                sync=False
            )
    def insert_data_to_milvus(self, df, user_id):
        '''插入数据到milvus中'''
        self.check_collections_state(user_id)
        data = df.to_dict('records')
        data_insert = []
        for line in data:
            hash_check = line['hash_check']
            if str(line['title']) == 'nan':
                line['title'] = ''
            if str(line['content']) == 'nan':
                line['content'] = ''
            if str(line['page_count']) == 'nan':
                line['page_count'] = ''
            if type(line['vector']) == str:
                line['vector'] = np.array(eval(line['vector']), dtype=np.float32)
            if isinstance(line['page_count'], int):
                line['page_count'] = str(line['page_count'])
            line['content'] = line['content'][:6666]
            res = self.client.query(
                collection_name=user_id,
                filter=f"hash_check=='{hash_check}'",
                output_fields=['hash_check']
            )
            if len(list(res)) != 0:
                continue
            else:
                data_insert.append(line)
        gr.Info(f'{len(data_insert)} insert to milvus!')
        if len(data_insert)!=0:
            res = self.client.insert(collection_name=user_id, data=data_insert)
            self.client.flush(
                collection_name=user_id
            )
            return pd.DataFrame(data_insert)
        else:
            return pd.DataFrame([])

    def get_collection_counts(self, user_id):
        '''获取milvus中指定集合中有多少条数据'''
        self.client.flush(collection_name=user_id)
        res = self.client.get_collection_stats(collection_name=user_id)
        print(res)

    def drop_collection(self, user_id):
        '''删除集合'''
        if self.client.has_collection(collection_name=user_id):
            self.client.drop_collection(collection_name=user_id)
            print(f'{user_id} collection delete!')
        else:
            gr.Warning(f'{user_id} collection not exist')

    def check_collections_state(self, user_id):
        '''检查集合是否加载到内存中'''
        collections_state = self.client.get_load_state(user_id)
        if collections_state['state'].name == 'NotLoad':
            self.client.load_collection(
                collection_name=user_id,
                replica_number=1
            )

    def search_vectors_single(self, user_id, article_ids, query_vector, limit=3):
        '''根据用户id，文章id和向量来搜索相关内容，仅限搜索一篇文章'''
        self.check_collections_state(user_id)

        res = self.client.search(
            collection_name=user_id,
            data=[query_vector],
            limit=limit,
            filter=f"article_id=='{article_ids}'",
            output_fields=['content', 'title', 'file_from', 'page_count']
        )
        output_info_list = self.get_search_info(res)
        return output_info_list

    def get_search_info(self, res):
        '''整理查询到结果'''
        output_info_list = []
        for res_i in list(res)[0]:
            info = {}
            info['title'] = res_i['entity']['title']
            info['content'] = res_i['entity']['content']
            info['page_count'] = res_i['entity']['page_count']
            info['file_from'] = res_i['entity']['file_from']
            output_info_list.append(info)
        return output_info_list

    def search_vectors_with_articles(self, user_id, query_vector, articles_id_list, limit=3):
        '''根据文章id列表来查询相似向量'''
        self.check_collections_state(user_id)
        filter_conditions = f"article_id in {articles_id_list}"
        res = self.client.search(
            collection_name=user_id,
            data=[query_vector],
            limit=limit,
            filter=filter_conditions,
            output_fields=['content', 'title', 'file_from', 'page_count']
        )
        output_info_list = self.get_search_info(res)
        return output_info_list

    def delete_data_by_article_id(self, user_id, articles_id_list):
        '''根据文章id删除数据'''
        filter_conditions = f"article_id in {articles_id_list}"
        res = self.client.delete(
            collection_name=user_id,
            filter=filter_conditions
        )
        self.client.flush(collection_name=user_id)
        print(f"{res['delete_count']} record has been delete!")

if __name__ == '__main__':
    from tqdm import tqdm
    from utils.tool import load_data, encrypt_username, read_json_file, save_json_file
    from utils.config_init import akb_conf_class
    manager = MilvusArticleManager()
    user_name = 'pandas'
    user_id = encrypt_username(user_name)
    manager.create_collection(user_id)
    # manager.drop_collection(user_id)
    manager.get_collection_counts(user_id)
    pandas_info_path = r'C:\use\code\RapidOcr_small\data\rag\data_csv\pandas'
    article_dict = {}
    for file_name in tqdm(os.listdir(pandas_info_path)):
        file_base, _ = os.path.splitext(file_name)
        csv_path = os.path.join(pandas_info_path, file_name)
        df = pd.read_csv(csv_path)
        article_id = df.iloc[10]['article_id']
        if article_id not in article_dict:
            article_dict[article_id] = file_base
    akb_conf_class.get_database_config('milvus')
    user_info = read_json_file(akb_conf_class.articles_user_path)
    user_info[user_name] = article_dict
    save_json_file(user_info, akb_conf_class.articles_user_path)
    print(user_info)
        # manager.insert_data_to_milvus(df, user_id)
    # manager.get_collection_counts(user_id)


    #         print(article_id)
    #         print(user_id)
    #         result_list = manager.search_vectors_with_articles(user_id, vector, [article_id])
    #         print(result_list[0])
    #         break

    # manager.delete_data_by_article_id(user_id, [article_ids_1])
    # result_list = manager.search_vectors(user_id, article_ids, vector)
    # print(result_list)
