import os

import ollama
import gradio as gr
from utils.tool import save_json_file, read_json_file

def get_all_model_name_and_path(model_dir = r'C:\use\model'):
    '''获取指定目标路径上的模型名字和具体的路径'''
    all_model_list = []
    try:
        res = ollama.list()
        ollama_model_list = []
        for model in res.models:
            ollama_model_list.append('ollama_'+model.model)
    except:
        gr.Warning('ollama not exist!')
        ollama_model_list = []
    local_model_dir = []
    local_model_name_path = {}
    for model_name in os.listdir(model_dir):
        model_name_i = 'local_'+model_name
        local_model_name_path[model_name_i] = os.path.join(model_dir, model_name)
        local_model_dir.append(model_name_i)
    all_model_list.extend(ollama_model_list)
    all_model_list.extend(local_model_dir)
    return all_model_list, local_model_name_path

def get_root_dir():
    return os.path.dirname(os.path.dirname(os.path.join(os.path.dirname(__file__))))

def save_all_model_config(
        data_base_type, max_history_len,
        default_model_dir, chat_model_type, embedding_model_type,
        max_output_len, rag_top_k, max_rag_len,
        chat_default_system,
        mysql_host, mysql_port, mysql_user, mysql_password, mysql_database, mysql_table,
        milvus_host, milvus_port,
        lancedb_data_dir,
        es_host, es_port, es_index_name, es_scheme,
        ollama_host, ollama_port,
        request: gr.Request
):
    '''对用户提交的配置进行保存'''
    project_root_dir = get_root_dir()
    all_config = {
        'database_type': ['lancedb', 'milvus', 'mysql', 'es'],
        'saved_chat_record_count': max_history_len,
        'default_model_dir': default_model_dir,
        'default_chat_model': chat_model_type,
        'default_embed_model': embedding_model_type,
        'all_model_list': [],
        'local_model_name_path_dict': {},
        'max_output_len': max_output_len,
        'rag_top_k': rag_top_k,
        'max_rag_len': max_rag_len,
        'caht_default_system': chat_default_system,
        'default_database_type':data_base_type,
        'mysql': {
            'host': mysql_host,
            'port': mysql_port,
            'user': mysql_user,
            'password': mysql_password,
            'database': mysql_database,
            'table': mysql_table,
            'id2article': os.path.join(project_root_dir, 'data', 'database', 'mysql', 'user_article_mapping.json'),
            'article_group': os.path.join(project_root_dir, 'data', 'database', 'mysql', 'kb_article_mappping.json')
        },
        'milvus': {
            'host': milvus_host,
            'port': milvus_port,
            'id2article': os.path.join(project_root_dir, 'data', 'database', 'milvus', 'user_article_mapping.json'),
            'article_group': os.path.join(project_root_dir, 'data', 'database', 'milvus', 'kb_article_mappping.json')
        },
        'lancedb': {
            'data_dir': lancedb_data_dir,
            'id2article': os.path.join(project_root_dir, 'data', 'database', 'lancedb', 'user_article_mapping.json'),
            'article_group': os.path.join(project_root_dir, 'data', 'database', 'lancedb', 'kb_article_mappping.json')
        },
        'es': {
            'host': es_host,
            'port': es_port,
            'index_name': es_index_name,
            'scheme': es_scheme,
            'id2article': os.path.join(project_root_dir, 'data', 'database', 'es', 'user_article_mapping.json'),
            'article_group': os.path.join(project_root_dir, 'data', 'database', 'es', 'kb_article_mappping.json')
        },
        'ollama':{
            'host': ollama_host,
            'port': ollama_port,
        }
    }
    all_model_list, local_model_name_path = get_all_model_name_and_path(default_model_dir)
    all_config['all_model_list'] = all_model_list
    all_config['local_model_name_path_dict'] = local_model_name_path
    user_name = request.username
    root_dir = os.path.dirname(os.path.dirname(os.path.join(os.path.dirname(__file__))))
    config_path = os.path.join(root_dir, 'config', 'config.yaml')
    if os.path.exists(config_path):
        old_config_info = read_json_file(config_path)
        old_config_info[user_name] = all_config
        save_json_file(old_config_info, config_path)
    else:
        save_config_info = {}
        save_config_info[user_name] = all_config
        save_json_file(save_config_info, config_path)
    gr.Info('save config successful')



def create_dir_if_not_exists(dir_path):
    os.makedirs(dir_path, exist_ok=True)

def get_mapping_paths(project_root_dir, db_name):
    id2article = os.path.join(project_root_dir, 'data', 'database', db_name, 'user_article_mapping.json')
    article_group = os.path.join(project_root_dir, 'data', 'database', db_name, 'kb_article_mappping.json')
    return id2article, article_group

def init_new_user_config(model_dir, lancedb_data_dir):
    '''初始化基础的默认配置'''
    all_model_list, local_model_name_path = get_all_model_name_and_path(model_dir)
    project_root_dir = get_root_dir()
    database_types = ['lancedb', 'milvus', 'mysql', 'es']
    config_dict = {
        'database_type': database_types,
        'saved_chat_record_count': 6,
        'all_model_list': all_model_list,
        'local_model_name_path': local_model_name_path,
        'max_output_len': 10240,
        'rag_top_k': 10,
        'max_rag_len': 10000,
        'caht_default_system': 'You are a helpful assistant.',
        'default_chat_model': 'ollama_qwen2.5:0.5b',
        'default_embed_model': 'ollama_bge-m3:latest',
        'default_model_dir': model_dir,
        'default_database_type': 'lancedb',
        'ollama':{
            'host':'127.0.0.1',
            'port':11434
        }
    }

    # 数据库相关配置
    db_configs = {
        'mysql': {
            'host': 'localhost',
            'port': 3310,
            'user': 'root',
            'password': 'root',
            'database': 'gradio_bot',
            'table': 'article_info',
            'id2article': os.path.join(project_root_dir, 'data', 'database', 'mysql', 'user_article_mapping.json'),
            'article_group': os.path.join(project_root_dir, 'data', 'database', 'mysql', 'kb_article_mappping.json')

        },
        'milvus': {
            'host': '127.0.0.1',
            'port': 19530,
            'id2article': os.path.join(project_root_dir, 'data', 'database', 'milvus', 'user_article_mapping.json'),
            'article_group': os.path.join(project_root_dir, 'data', 'database', 'milvus', 'kb_article_mappping.json')
        },
        'lancedb': {
            'data_dir': lancedb_data_dir,
            'id2article': os.path.join(project_root_dir, 'data', 'database', 'lancedb', 'user_article_mapping.json'),
            'article_group': os.path.join(project_root_dir, 'data', 'database', 'lancedb', 'kb_article_mappping.json')
        },
        'es': {
            'host': 'localhost',
            'port': 9200,
            'index_name': 'article_index',
            'scheme': 'http',
            'id2article': os.path.join(project_root_dir, 'data', 'database', 'es', 'user_article_mapping.json'),
            'article_group': os.path.join(project_root_dir, 'data', 'database', 'es', 'kb_article_mappping.json')
        }
    }

    for db_name in database_types:
        id2article, article_group = get_mapping_paths(project_root_dir, db_name)
        db_configs[db_name]['id2article'] = id2article
        db_configs[db_name]['article_group'] = article_group

    config_dict.update(db_configs)
    return config_dict

def init_data_dir():
    root_dir = get_root_dir()
    # 配置文件路径
    config_path = os.path.join(root_dir, 'config', 'config.yaml')
    create_dir_if_not_exists(os.path.dirname(config_path))
    # 模型目录
    model_dir = os.path.join(root_dir, 'model')
    create_dir_if_not_exists(model_dir)
    # 数据库目录
    database_dir = os.path.join(root_dir, 'data', 'database')
    # 各数据库子目录
    for db_name in ['lancedb', 'es', 'milvus', 'mysql']:
        db_dir = os.path.join(database_dir, db_name)
        create_dir_if_not_exists(db_dir)
        if db_name == 'lancedb':
            lancedb_data_dir = os.path.join(db_dir, 'data')
            create_dir_if_not_exists(lancedb_data_dir)

    return model_dir, lancedb_data_dir, config_path

def demo_config(request:gr.Request):
    '''第一次加载配置'''
    model_dir, lancedb_data_dir, config_path = init_data_dir()
    if not os.path.exists(config_path):
        config_dict = init_new_user_config(model_dir, lancedb_data_dir)
    else:
        config_dict_all = read_json_file(config_path)
        user_name = request.username
        if user_name not in config_dict_all:
            config_dict = init_new_user_config(model_dir, lancedb_data_dir)
        else:
            config_dict = config_dict_all[user_name]
    return config_dict

def config_ui_show():
    '''默认基配置的界面'''
    with gr.Group():
        all_config = gr.JSON(visible=False)
        demo.load(demo_config, inputs=None, outputs=all_config)
        with gr.Row():
            data_base_type = gr.Dropdown(choices=[], label='选用的数据库', interactive=True)
            max_history_len = gr.Number(value=0, label='保存的聊天记录数', interactive=True)
        with gr.Row():
            default_model_dir = gr.Textbox(value='', label='模型基本路径', interactive=True)
            chat_model_type = gr.Dropdown(choices=[], label="默认聊天模型", interactive=True, filterable=True)
            embedding_model_type = gr.Dropdown(choices=[], label='默认embedding模型选择', interactive=True)
        with gr.Row():
            max_output_len = gr.Number(value=0, label='生成模型的生成长度', interactive=True)
            rag_top_k = gr.Number(value=0, label='rag查询返回的记录数', interactive=True)
            max_rag_len = gr.Number(value=0, label='挂载rag的最大长度', interactive=True)
        with gr.Row():
            chat_default_system = gr.Textbox(value='', interactive=True)
        with gr.Row():
            mysql_host = gr.Textbox(value='localhost', label='mysql的地址', interactive=True)
            mysql_port = gr.Number(value=3310, label='mysql的端口號', interactive=True)
            mysql_user = gr.Textbox(value='root', label='mysql的用戶', interactive=True)
            mysql_password = gr.Textbox(type='password', label='mysql用戶密碼', value='root', interactive=True)
            mysql_database = gr.Textbox(value='gradio_bot', label='mysql数据库名', interactive=True)
            mysql_table = gr.Textbox(value='article_info', label='mysql数据表名', interactive=True)
        with gr.Row():
            milvus_host = gr.Textbox(value='127.0.0.1', label='milvus地址', interactive=True)
            milvus_port = gr.Number(value=19530, label='milvus端口号', interactive=True)
        with gr.Row():
            lancedb_data_dir = gr.Textbox(value='', label='lancedb数据保存路径', interactive=True)
        with gr.Row():
            es_host = gr.Textbox(value='localhost', label='es的地址', interactive=True)
            es_port = gr.Number(value=9200, label='es的端口号', interactive=True)
            es_index_name = gr.Textbox(value='article_index', label='es的index名字', interactive=True)
            es_scheme = gr.Textbox(value='http', label='es连接时使用的协议', interactive=True)
        with gr.Row():
            ollama_host = gr.Textbox(value='127.0.0.1', label='ollama的地址', interactive=True)
            ollama_port = gr.Number(value=11434, label='ollama端口号', interactive=True)
        with gr.Row():
            save_button = gr.Button(value='保存配置')


    @all_config.change(
        inputs=all_config,
        outputs=[
            data_base_type, max_history_len,
            default_model_dir, chat_model_type, embedding_model_type,
            max_output_len, rag_top_k, max_rag_len,
            chat_default_system,
            mysql_host, mysql_port,mysql_user,mysql_password, mysql_database, mysql_table,
            milvus_host,milvus_port,
            lancedb_data_dir,
            es_host, es_port, es_index_name, es_scheme,
            ollama_host, ollama_port
        ]
    )
    def all_config_load(all_config):
        '''将本地配置映射到界面上'''
        data_base_type = gr.Dropdown(choices=all_config['database_type'], label='选用的数据库', value=all_config['default_database_type'],
                                     interactive=True)
        max_history_len = gr.Number(value=all_config['saved_chat_record_count'], label='保存的聊天记录数',
                                    interactive=True)
        default_model_dir = gr.Textbox(value=all_config['default_model_dir'], label='模型基本路径')
        chat_model_type = gr.Dropdown(choices=all_config['all_model_list'], label="默认聊天模型",
                                      interactive=True, filterable=True, value=all_config['default_chat_model'])
        embedding_model_type = gr.Dropdown(choices=all_config['all_model_list'], label='默认embedding模型选择',
                                           interactive=True, value=all_config['default_embed_model'])
        max_output_len = gr.Number(value=all_config['max_output_len'], label='生成模型的生成长度',
                                   interactive=True)
        rag_top_k = gr.Number(value=all_config['rag_top_k'], label='rag查询返回的记录数', interactive=True)
        max_rag_len = gr.Number(value=all_config['max_rag_len'], label='挂载rag的最大长度', interactive=True)
        chat_default_system = gr.Textbox(value=all_config['caht_default_system'], interactive=True)

        '''mysql配置'''
        mysql_host = gr.Textbox(value=all_config['mysql']['host'], label='mysql的地址')
        mysql_port = gr.Number(value=all_config['mysql']['port'], label='mysql的端口號')
        mysql_user = gr.Textbox(value=all_config['mysql']['user'], label='mysql的用戶')
        mysql_password = gr.Textbox(type='password', label='mysql用戶密碼', value=all_config['mysql']['password'])
        mysql_database = gr.Textbox(value=all_config['mysql']['database'], label='mysql数据库名')
        mysql_table = gr.Textbox(value=all_config['mysql']['table'], label='mysql数据表名')

        '''milvus配置'''
        milvus_host = gr.Textbox(value=all_config['milvus']['host'], label='milvus地址')
        milvus_port = gr.Number(value=all_config['milvus']['port'], label='milvus端口号')

        '''lancedb配置'''
        lancedb_data_dir = gr.Textbox(value=all_config['lancedb']['data_dir'], label='lancedb数据保存路径')

        '''es配置'''
        es_host = gr.Textbox(value=all_config['es']['host'], label='es的地址')
        es_port = gr.Number(value=all_config['es']['port'], label='es的端口号')
        es_index_name = gr.Textbox(value=all_config['es']['index_name'], label='es的index名字')
        es_scheme = gr.Textbox(value=all_config['es']['scheme'], label='es连接时使用的协议')

        '''ollama配置'''
        ollama_host = gr.Textbox(value='127.0.0.1', label='ollama的地址', interactive=True)
        ollama_port = gr.Number(value=11434, label='ollama端口号', interactive=True)
        return (data_base_type, max_history_len,
                default_model_dir, chat_model_type, embedding_model_type,
            max_output_len, rag_top_k, max_rag_len,
            chat_default_system,
            mysql_host, mysql_port,mysql_user,mysql_password, mysql_database, mysql_table,
            milvus_host,milvus_port,
            lancedb_data_dir,
            es_host, es_port, es_index_name, es_scheme,
            ollama_host, ollama_port
                )

    save_button.click(
        save_all_model_config,
        inputs=[
            data_base_type, max_history_len,
            default_model_dir, chat_model_type, embedding_model_type,
            max_output_len, rag_top_k, max_rag_len,
            chat_default_system,
            mysql_host, mysql_port,mysql_user,mysql_password, mysql_database, mysql_table,
            milvus_host,milvus_port,
            lancedb_data_dir,
            es_host, es_port, es_index_name, es_scheme,
            ollama_host, ollama_port
        ],
        outputs=None
    )

    @default_model_dir.change(
        inputs=default_model_dir,
        outputs=[chat_model_type, embedding_model_type]
    )
    def default_model_dir_change(input_value):
        '''当模型路径改变时，实时提取模型目录'''
        if os.path.exists(input_value):
            all_model_list, _ = get_all_model_name_and_path(model_dir=input_value)
            chat_model_type = gr.Dropdown(choices=all_model_list, label="默认聊天模型", interactive=True, filterable=True)
            embedding_model_type = gr.Dropdown(choices=all_model_list, label='默认embedding模型选择', interactive=True)
        else:
            gr.Warning('路径设置不正确')
            chat_model_type = gr.Dropdown(choices=[], label="默认聊天模型", interactive=True,
                                          filterable=True)
            embedding_model_type = gr.Dropdown(choices=[], label='默认embedding模型选择', interactive=True)
        return chat_model_type, embedding_model_type

if __name__ == '__main__':
    '''
    我现在的逻辑是配制有改动就保存到本地中。
    '''
    with gr.Blocks() as demo:
        config_ui_show()
    demo.launch(
        auth=[('a', "a"), ('b', 'b'), ('pandas', '123')],
    )