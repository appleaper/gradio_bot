import os
import time
import pandas as pd
import torch
import ollama
import lancedb
import gradio as gr
import numpy as np
from ollama import chat
from gradio import ChatMessage
from threading import Thread
from transformers import TextIteratorStreamer

from not_use.gradio.Helper.上传文件 import output_file_1
from utils.tool import read_user_info_dict, reverse_dict
from local.local_api import load_model_cached, load_rag_cached
from utils.tool import encrypt_username
from local.database.milvus.milvus_article_management import MilvusArticleManager
from utils.config_init import rag_top_k, max_history_len, max_rag_len, qwen_support_list, ollama_support_list, device_str, akb_conf_class
from local.rag.online_search_capability import online_search
from local.rag.search_data_from_database import get_info_from_mysql
from local.rag.search_data_from_database import get_info_from_es

def add_rag_info(textbox, book_type, database_type, top_k, user_name):
    '''
    暂时去掉了模型审核内容是否相关的部分
    :param textbox: 用户提问
    :param book_type: 知识库的名字
    :param database_type: 数据库类型
    :param top_k: 取多少条相关记录
    :return: rag文字
    '''
    # model, tokenizer = load_model_cached('qwen2.5-0.5B-Instruct')
    akb_conf_class.get_database_config(database_type)
    all_knowledge_bases_record = read_user_info_dict(user_name, akb_conf_class.kb_article_map_path)
    inverted_dict = reverse_dict(read_user_info_dict(user_name, akb_conf_class.articles_user_path))


    if akb_conf_class.database_type=='lancedb':
        rag_model, rag_tokenizer = load_rag_cached('bge_m3')
        vector = rag_model.encode(textbox, batch_size=1, max_length=8192)['dense_vecs']
        db = lancedb.connect(akb_conf_class.lancedb_data_dir)
        records_df = pd.DataFrame()
        for table_name in all_knowledge_bases_record[book_type]:
            tb = db.open_table(inverted_dict[table_name])
            records = tb.search(vector).limit(top_k).to_pandas()
            records_df = pd.concat((records, records_df))
        sorted_records_df = records_df.sort_values(by='_distance').iloc[:top_k]
    elif akb_conf_class.database_type == 'milvus':
        rag_model, rag_tokenizer = load_rag_cached('bge_m3')
        vector = rag_model.encode(textbox, batch_size=1, max_length=8192)['dense_vecs']
        user_id = encrypt_username(user_name)
        manager = MilvusArticleManager()
        articles_id_list = list(inverted_dict.values())
        vector = np.array(vector, dtype=np.float32)
        res = manager.search_vectors_with_articles(user_id, vector, articles_id_list, limit=rag_top_k)
        sorted_records_df = pd.DataFrame(res)
    elif akb_conf_class.database_type=='mysql':
        sorted_records_df = get_info_from_mysql(textbox, book_type, top_k, user_name)
    elif akb_conf_class.database_type=='es':
        sorted_records_df = get_info_from_es(textbox, book_type, top_k, user_name)
    else:
        raise gr.Error(f'{akb_conf_class.database_type} not support!')
    rag_str = ''
    now_str_count = 0
    for record_index, record in sorted_records_df.iterrows():
        if 'tile' in record and 'content' in record:
            rag_str_i = f'相关文档{record_index}:\n标题:{record["title"]}\n内容:{record["content"]}\n\n'
        else:
            rag_str_i = ''
        if now_str_count < max_rag_len:
            rag_str += rag_str_i
            now_str_count += len(rag_str_i)
        else:
            rag_str += rag_str_i[:(max_rag_len - now_str_count)]
            break

        # messages = [
        #     {"role":"system","content":'你是一名评估员，负责评估检索到的文档与用户问题的相关性。如果文档包含与用户问题相关的关键词或语义含义，'
        #                                '将其评定为相关。这不需要是一个严格的测试。目标是筛选出错误的检索结果。给出一个二元分数“是”或“否”，以表明该文档是否与问题相关。'},
        #     {'role':'user', 'content':f'相关文档：{rag_str},用户提问:{textbox}'}
        # ]
        # response_message = qwen_model_detect(messages, model, tokenizer)
        # if response_message == '是':
        #     if now_str_count < max_rag_len:
        #         rag_str += rag_str_i
        #         now_str_count += len(rag_str_i)
        #     else:
        #         rag_str += rag_str_i[:(max_rag_len - now_str_count)]
        #         break
    # print(rag_str)
    return rag_str

def insert_role_setting(history, system_state):
    '''添加角色设定和设定和使用的聊天记录数'''
    if len(history) == 0:
        history = [{"role":"system","content":system_state}]
    elif len(history) >= max_history_len:
        history = history[-max_history_len:]
        history[0] = {"role": "system", "content": system_state}
    else:
        history[0] = {"role":"system","content":system_state}
    return history

def load_rag_system(user_ask, book_type, database_type, user_name, is_connected_network, history, system_state):
    '''挂载rag'''
    if str(book_type) == 'None':
        rag_str = ''
    else:
        rag_str = add_rag_info(user_ask, book_type, database_type, rag_top_k, user_name)
    if is_connected_network:
        online_search_info_str = online_search(user_ask)
        rag_str += online_search_info_str
    insert_role_setting(history, system_state)
    if len(rag_str) == 0:
        history.append(
            {'role':'user', 'content':f'{user_ask}'}
        )
    else:
        history.append(
            {'role':'user', 'content':f'相关文档：{rag_str},以上的参考文档只是相关性，参考时需要考虑其是否正确。请回答以下用户提问:{user_ask}'}
        )
    return history

def deal_deepseek():
    think_str = ''
    start_time = time.time()
    think_flag = False
    begin_of_sentence_flag = False

    if model_name.startswith('deepseek'):
        response_think = ChatMessage(
            content="",
            metadata={"title": "_Thinking_ step-by-step", "id": 0, "status": "pending"}
        )
        yield response_think
    else:
        response_think = None
    if model_name.startswith('deepseek'):
        if '<｜begin▁of▁sentence｜>' in output:
            begin_of_sentence_flag = True
        else:
            if '<think>' in output:
                begin_of_sentence_flag = False
            if '<think>' in output and begin_of_sentence_flag == False:
                temp_list = output.split('<think>', 1)
                think_str += temp_list[1]
                response_think.content = think_str
                think_flag = True
                # begin_of_sentence_flag = False
                yield response_think
            elif '</think>' in output and begin_of_sentence_flag == False:
                think_flag = False
                str_list = output.split('</think>', 1)
                think_str += str_list[0]
                response_str += str_list[1]
                response_think.content = think_str
                response_think.metadata["status"] = "done"
                response_think.metadata["duration"] = time.time() - start_time
                yield response_think
            else:
                if think_flag == False and begin_of_sentence_flag == False:
                    response_str += output
                    response = [
                        response_think,
                        ChatMessage(
                            content=response_str
                        )
                    ]
                    yield response
                elif think_flag == True and begin_of_sentence_flag == False:
                    think_str += output
                    response_think.content = think_str
                    yield response_think
                else:
                    yield ''


# textbox, chatbot, system_input, history_state, model_type, model_name, book_type, chat_database_type, is_connected_network
def local_chat(
        textbox,
        history,
        system_state,
        model_type,
        parm_b,
        book_type,
        database_type,
        is_connected_network,
        request: gr.Request
):
    '''

    :param textbox: 用户输入，可能带了其他东西
    :param history: 历史记录
    :param system_state: 角色设定
    :param model_type: 模型类别
    :param parm_b: 模型名字
    :param book_type: 知识库的名字
    :param database_type 数据库名字
    :param is_connected_network: 是否联网搜索
    :param request: 主要是用来获取用户名字
    :return:
    '''
    user_name = request.username
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model_name = model_type + '-' + parm_b
    user_ask = textbox['text']
    user_upload_file = textbox['files']
    history = load_rag_system(user_ask, book_type, database_type, user_name, is_connected_network, history, system_state)
    if model_name in qwen_support_list:
        model, tokenizer = load_model_cached(model_name)
        conversion = tokenizer.apply_chat_template(history, add_generation_prompt=True, tokenize=False)
        model_inputs = tokenizer(conversion, return_tensors="pt").to(device_str)
        streamer = TextIteratorStreamer(tokenizer)
        generation_kwargs = dict(model_inputs, streamer=streamer, max_new_tokens=512)
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        response_str = ''
        for new_text in streamer:
            output = new_text.replace(conversion, '')
            if output:
                if model_name.startswith('qwen'):
                    if output.endswith('<|im_end|>'):
                        output = output.replace('<|im_end|>', '')
                    response_str += output
                    yield response_str
                else:
                    yield output
    elif model_name in ollama_support_list:
        stream = chat(
            model=parm_b,
            messages=history,
            stream=True,
        )
        output_str = ''
        for chunk in stream:
            output_str += chunk['message']['content']
            yield output_str
    else:
        gr.Error(f'{model_name} not support')
        assert False, 'model name not support!'