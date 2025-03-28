import os
import pymysql
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.tool import singleton
from utils.config_init import akb_conf_class


@singleton
class MySQLDatabase:
    def __init__(self, host, user, password, port):
        self.host = host
        self.user = user
        self.port = port
        self.password = password
        self.connection = None

    def connect(self):
        '''连接数据库'''
        try:
            self.connection = pymysql.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                port=self.port,  # 使用默认端口 3306
                charset='utf8mb4'  # 指定字符集为 utf8mb4
            )
            print('成功连接到 MySQL 服务器')
            with self.connection.cursor() as cursor:
                cursor.execute('SHOW VARIABLES LIKE \'character_set_connection\';')
                result = cursor.fetchone()
        except pymysql.Error as e:
            print(f"连接错误: {e}")

    def check_table_charset(self, table_name):
        if self.connection is None:
            print("请先连接到数据库")
            return
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(f"SHOW CREATE TABLE {table_name};")
                result = cursor.fetchone()
                create_table_statement = result[1]
                print(f"表 {table_name} 的创建语句: {create_table_statement}")
        except pymysql.Error as e:
            print(f"查询表字符集时出错: {e}")

    def create_database(self, database_name):
        '''创建数据库'''
        if self.connection is None:
            self.connect()
        try:
            with self.connection.cursor() as cursor:
                # 创建数据库时指定字符集为 utf8mb4
                create_db_query = f"CREATE DATABASE IF NOT EXISTS {database_name} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
                cursor.execute(create_db_query)
            self.connection.commit()
            print(f"数据库 {database_name} 创建成功")
        except pymysql.Error as e:
            print(f"创建数据库时出错: {e}")

    def use_database(self, database_name):
        '''选择数据库'''
        if self.connection is None:
            print("请先连接到 MySQL 服务器")
            return
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(f"USE {database_name}")
            self.connection.commit()
            print(f'成功使用数据库 {database_name}')
        except pymysql.Error as e:
            print(f"使用数据库 {database_name} 时出错: {e}")


    def delete_database(self, database_name):
        '''删除数据库'''
        if self.connection is None:
            self.connect()
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(f"DROP DATABASE IF EXISTS {database_name}")
            self.connection.commit()
            print(f"数据库 {database_name} 删除成功")
        except pymysql.Error as e:
            print(f"删除数据库时出错: {e}")


    def connect_to_database(self, database_name):
        '''连接数据库'''
        try:
            self.connection = pymysql.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=database_name,
                charset='utf8mb4'  # 指定字符集为 utf8mb4
            )
            print(f'成功连接到数据库 {database_name}')
        except pymysql.Error as e:
            print(f"连接到数据库 {database_name} 时出错: {e}")

    def create_table(self, table_name):
        '''创建数据表'''
        if self.connection is None:
            print("请先连接到数据库")
            return
        try:
            with self.connection.cursor() as cursor:
                columns = "article_id VARCHAR(255), user_id VARCHAR(255), title VARCHAR(255), content TEXT, page_count VARCHAR(255), file_from VARCHAR(255), hash_check VARCHAR(255), PRIMARY KEY (hash_check)"
                # 创建表时指定字符集和排序规则
                create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns}) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
                cursor.execute(create_table_query)
            self.connection.commit()
            print(f"数据表 {table_name} 创建成功")
        except pymysql.Error as e:
            print(f"创建数据表时出错: {e}")

    def _replace_nan(self, value):
        if isinstance(value, float) and np.isnan(value):
            return None
        return value

    def insert_data(self, table_name, article_id, user_id, title, content, page_count, file_from, hash_check):
        if self.connection is None:
            print("请先连接到数据库")
            return

        values = [article_id, user_id, title, content, page_count, file_from, hash_check]
        values = [self._replace_nan(val) for val in values]

        try:
            with self.connection.cursor() as cursor:
                insert_query = f"INSERT INTO {table_name} (article_id, user_id, title, content, page_count, file_from, hash_check) VALUES (%s, %s, %s, %s, %s, %s, %s)"
                # 打印实际执行的插入语句
                # debug_query = insert_query.replace("%s", "'{}'").format(*values)
                # print(f"执行的插入语句: {debug_query}")
                cursor.execute(insert_query, tuple(values))
            self.connection.commit()
            return 1
        except pymysql.Error as e:
            print(f"插入数据时出错: {e}")
            print(f"执行的 SQL 语句: {insert_query % tuple(values)}")
            return 0

    def select_data(self, table_name, user_id, input_text, article_id_list, limit=3):
        '''查询数据'''
        if self.connection is None:
            print("请先连接到数据库")
            return
        try:
            with self.connection.cursor() as cursor:
                placeholders = ', '.join(['%s'] * len(article_id_list))
                select_query = f"SELECT * FROM {table_name} WHERE user_id = %s AND article_id IN ({placeholders}) AND (LOWER(title) LIKE %s OR LOWER(content) LIKE %s) LIMIT %s"
                search_pattern = f"%{input_text.lower()}%"
                params = (user_id, *article_id_list, search_pattern, search_pattern, limit)
                print("执行的 SQL 语句:", select_query)
                cursor.execute(select_query, params)
                results = cursor.fetchall()
                return results
        except pymysql.Error as e:
            print(f"查询数据时出错: {e}")

    def count_table_data(self, table_name):
        """查询指定表中的数据数量"""
        if self.connection is None:
            print("请先连接到数据库")
            return
        try:
            with self.connection.cursor() as cursor:
                count_query = f"SELECT COUNT(*) FROM {table_name}"
                cursor.execute(count_query)
                result = cursor.fetchone()
                if result:
                    data_count = result[0]
                    print(f"表 {table_name} 中的数据数量为: {data_count}")
                else:
                    print("未获取到数据数量。")
        except pymysql.Error as e:
            print(f"查询数据数量时出错: {e}")

    def close_connection(self):
        '''关闭数据库连接'''
        if self.connection:
            self.connection.close()
            print("数据库连接已关闭")

    def insert_data_format_df(self, table_name, df):
        '''插入格式是'''
        insert_list = []
        for index, article_info in tqdm(df.iterrows(), total=len(df)):
            user_id = article_info['user_id']
            article_id = article_info['article_id']
            title = self._replace_nan(article_info['title'])
            content = self._replace_nan(article_info['content'])
            page_count = self._replace_nan(article_info['page_count'])
            file_from = self._replace_nan(article_info['file_from'])
            hash_check = article_info['hash_check']
            check_flag = self.insert_data(
                table_name, article_id, user_id, title, content, page_count, file_from, hash_check
            )
            if check_flag:
                insert_list.append(article_info)
        return pd.DataFrame(insert_list)

    def delete_data_by_user_and_article_ids(self, table_name, user_id, article_id_list):
        """
        根据用户 ID 和文章列表 ID 删除数据
        :param table_name: 表名
        :param user_id: 用户 ID
        :param article_id_list: 文章 ID 列表
        """
        if self.connection is None:
            print("请先连接到数据库")
            return
        try:
            placeholders = ', '.join(['%s'] * len(article_id_list))
            delete_query = f"DELETE FROM {table_name} WHERE user_id = %s AND article_id IN ({placeholders})"
            params = (user_id, *article_id_list)
            with self.connection.cursor() as cursor:
                cursor.execute(delete_query, params)
            self.connection.commit()
            print("数据删除成功")
            return True
        except pymysql.Error as e:
            print(f"删除数据时出错: {e}")
            return False


def init_mysql():
    '''插入数据的时候，初始化mysql'''
    manager = MySQLDatabase(
        host=akb_conf_class.mysql_host,
        user=akb_conf_class.mysql_user,
        password=akb_conf_class.mysql_password,
        port=akb_conf_class.mysql_port
    )
    manager.connect()
    manager.create_database(akb_conf_class.mysql_database_name)
    manager.use_database(akb_conf_class.mysql_database_name)
    return manager

# 以下是使用示例
if __name__ == "__main__":
    from utils.tool import save_rag_group_name, read_json_file
    # from utils.config_init import akb_conf_class
    from utils.config_init import rag_data_csv_dir
    database_name = 'gradio_bot'
    table_name = 'article_info'

    db = MySQLDatabase('localhost', 'root', 'root', port=3310)
    db.connect()
    db.create_database(database_name)
    db.use_database(database_name)

    # db.check_table_charset(table_name)
    # db.delete_table(table_name)
    db.create_table(table_name)

    user_name = 'pandas'
    akb_conf_class.get_database_config('mysql')
    akb_conf_class.init_article_user_and_kb_mapping_file()
    article_info_dir = os.path.join(rag_data_csv_dir, user_name)
    for filename in os.listdir(article_info_dir):
        article_path = os.path.join(article_info_dir, filename)
        df = pd.read_csv(article_path, encoding='utf8')
        data = read_json_file(akb_conf_class.articles_user_path)
        article_name, article_suffix = os.path.splitext(os.path.basename(article_path))
        article_id = save_rag_group_name(user_name, article_name, data, akb_conf_class.articles_user_path)
        for index, article_info in tqdm(df.iterrows(), total=len(df)):
            user_id = article_info['user_id']
            # article_id = article_info['article_id']
            title = article_info['title']
            content = article_info['content']
            page_count = article_info['page_count']
            file_from = article_info['file_from']
            hash_check = article_info['hash_check']
            db.insert_data(
                table_name, article_id, user_id, title, content, page_count, file_from, hash_check
            )

    # db.select_data(table_name, 'ae4863bd', '决策')
    db.count_table_data(table_name)
    # db.delete_database(database_name)
    # db.close_connection()
