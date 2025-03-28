import os

import pandas as pd
import torch
import ollama
import lancedb
import gradio as gr
import numpy as np
from ollama import chat

from threading import Thread
from transformers import TextIteratorStreamer
from utils.tool import read_user_info_dict, reverse_dict
from local.local_api import load_model_cached, load_rag_cached
from utils.tool import encrypt_username
from local.database.milvus.milvus_article_management import MilvusArticleManager
from utils.config_init import rag_top_k, max_history_len, max_rag_len, qwen_support_list, ollama_support_list, device_str, akb_conf_class
from local.rag.online_search_capability import online_search

def add_rag_info(textbox, book_type, rag_model, database_type, top_k, user_name):
    '''
    暂时去掉了模型审核内容是否相关的部分
    :param textbox: 用户提问
    :param book_type: 知识库
    :param rag_model: rag向量模型
    :param database_type: 数据库类型
    :param top_k: 取多少条相关记录
    :return: rag文字
    '''
    # model, tokenizer = load_model_cached('qwen2.5-0.5B-Instruct')
    akb_conf_class.get_database_config(database_type)
    all_knowledge_bases_record = read_user_info_dict(user_name, akb_conf_class.kb_article_map_path)
    inverted_dict = reverse_dict(read_user_info_dict(user_name, akb_conf_class.articles_user_path))
    vector = rag_model.encode(textbox, batch_size=1, max_length=8192)['dense_vecs']

    if akb_conf_class.database_type=='lancedb':
        db = lancedb.connect(akb_conf_class.lancedb_data_dir)
        records_df = pd.DataFrame()
        for table_name in all_knowledge_bases_record[book_type]:
            tb = db.open_table(inverted_dict[table_name])
            records = tb.search(vector).limit(top_k).to_pandas()
            records_df = pd.concat((records, records_df))
        sorted_records_df = records_df.sort_values(by='_distance').iloc[:top_k]
    elif akb_conf_class.database_type == 'milvus':
        user_id = encrypt_username(user_name)
        manager = MilvusArticleManager()
        articles_id_list = list(inverted_dict.values())
        vector = np.array(vector, dtype=np.float32)
        res = manager.search_vectors_with_articles(user_id, vector, articles_id_list, limit=rag_top_k)
        sorted_records_df = pd.DataFrame(res)
    else:
        raise gr.Error(f'{akb_conf_class.database_type} not support!')
    rag_str = ''
    now_str_count = 0
    for record_index, record in sorted_records_df.iterrows():
        rag_str_i = f'' \
                   f'相关文档{record_index}:\n' \
                   f'标题:{record["title"]}\n' \
                   f'内容:{record["content"]}\n' \
                   f'来源:{os.path.basename(record["file_from"])}的第{record["page_count"]}页/行\n\n'

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

# textbox, chatbot, system_input, history_state, model_type, model_name, book_type, chat_database_type, is_connected_network
def local_chat(
        textbox,
        show_history,
        system_state,
        history,
        model_type,
        parm_b,
        book_type,
        database_type,
        is_connected_network,
        request: gr.Request
):
    '''

    :param textbox: 用户提问
    :param show_history: 聊天组件
    :param system_state: 角色设定
    :param history: 历史记录
    :param model_type: 模型类别
    :param parm_b: 模型名字
    :param steam_check_box: 流式输出与否，str类型
    :param book_type: 知识库的名字
    :param is_connected_network: 是否联网搜索
    :param request: 当前登录用户的名字
    :return:
    '''
    user_name = request.username
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model_name = model_type + '-' + parm_b
    if str(book_type) == 'None':
        rag_str = ''
    else:
        rag_model, rag_tokenizer = load_rag_cached('bge_m3')
        rag_str = add_rag_info(textbox, book_type, rag_model, database_type, rag_top_k, user_name)
    if is_connected_network:
        online_search_info_str = online_search(textbox)
        rag_str += online_search_info_str
    if show_history is None:
        history = []
    if len(history) == 0:
        history = [{"role":"system","content":system_state}]
    elif len(history) >= max_history_len:
        history = history[-max_history_len:]
        history[0] = {"role": "system", "content": system_state}
    else:
        history[0] = {"role":"system","content":system_state}
    history.append(
        {'role':'user', 'content':f'相关文档：{rag_str},以上的参考文档只是相关性，参考时需要考虑其是否正确。请回答以下用户提问:{textbox}'}
    )
    if model_type != 'ollama':
        model, tokenizer = load_model_cached(model_name)
    else:
        model, tokenizer = None, None
    if model_name in qwen_support_list:
        conversion = tokenizer.apply_chat_template(history, add_generation_prompt=True, tokenize=False)
        model_inputs = tokenizer(conversion, return_tensors="pt").to(device_str)
        streamer = TextIteratorStreamer(tokenizer)
        generation_kwargs = dict(model_inputs, streamer=streamer, max_new_tokens=512)
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        response = ''
        show_history.append(())
        for new_text in streamer:
            output = new_text.replace(conversion, '')
            if output:
                if output.endswith('<|im_end|>'):
                    output = output.replace('<|im_end|>', '')
                    history.append({'role': 'assistant', 'content': response})
                show_history[-1] = (textbox,response)
                response += output
                yield '', show_history, history
    elif model_name in ollama_support_list:
        stream = chat(
            model=parm_b,
            messages=history,
            stream=True,
        )
        show_history.append(())
        output_str = ''
        for chunk in stream:
            response = chunk['message']['content']
            show_history[-1] = (textbox, output_str)
            output_str += response
            yield '', show_history, []
    else:
        gr.Error(f'{model_name} not support')
        assert False, 'model name not support!'