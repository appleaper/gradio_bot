import os.path

import torch
import lancedb
import hashlib
import functools
import gradio as gr
from local.qwen.qwen_api import qwen_model_init,qwen_model_detect
from local.llama3.llama3_api import llama3_model_init, llama3_model_detect
from local.MiniCPM.minicpm_api import minicpm_model_init, minicpm_model_detect
from config import conf_yaml
from database_data.emb_model.init_model_bge_m3 import model_init as bge_m3_model_init
from database_data.emb_model.init_model_bge_m3 import model_detect as bge_m3_model_detect
from threading import Thread
from transformers import TextIteratorStreamer
from local.rag.util import read_rag_name_dict

name2path = conf_yaml['local_chat']['name2path']
max_history_len = conf_yaml['local_chat']['max_history_len']
rag_database_name = conf_yaml['rag']['rag_database_name']
rag_top_k = conf_yaml['rag']['top_k']
rag_list_config_path = conf_yaml['rag']['rag_list_config_path']
bge_mdoel_path = conf_yaml['rag']['beg_model_path']

qwen_support_list = [
    'qwen2.5-7B-Instruct',
    'qwen2.5-7B-Instruct-AWQ',
    'qwen2.5-1.5-Instruct-Coder',
    'qwen2.5-7B-Coder',
    'qwen2-1.5B-Instruct',
    'qwen2-7B-Instruct-AWQ'
]
def load_model(model_name):
    if model_name in qwen_support_list:
        model, tokenizer = qwen_model_init(name2path[model_name])
    elif model_name == 'llama3-8b':
        model, tokenizer = llama3_model_init(name2path['llama3-8b'])
    elif model_name == 'MiniCPM3-4B':
        model, tokenizer = minicpm_model_init(name2path['MiniCPM3-4B'])
    else:
        gr.Error(f'{model_name} not support!')
        assert False, 'model name not support!'
    return model, tokenizer

def load_rag_model(model_name):
    if model_name == 'bge_m3':
        model, tokenizer = bge_m3_model_init(bge_mdoel_path)
        return model, tokenizer
    else:
        assert False, 'rag model not support!'

# 创建一个缓存装饰器
def cached_model_loader(func):
    cache = {}
    @functools.wraps(func)
    def wrapper(model_path):
        # 计算输入的哈希值
        input_hash = hashlib.md5(model_path.encode()).hexdigest()
        if input_hash not in cache:
            # 如果输入的哈希值不在缓存中，则加载模型并缓存结果
            cache[input_hash] = func(model_path)
        return cache[input_hash]
    return wrapper

# 使用缓存装饰器装饰模型加载函数
@cached_model_loader
def load_model_cached(model_name):
    return load_model(model_name)

@cached_model_loader
def load_rag_cached(model_name):
    return load_rag_model(model_name)

def add_rag_info(textbox, book_type, rag_model, rag_tokenizer, database_name, top_k):
    vector = rag_model.encode(textbox, batch_size=1, max_length=8192)['dense_vecs']
    db = lancedb.connect(database_name)

    data = read_rag_name_dict(rag_list_config_path)
    inverted_dict = {value: key for key, value in data.items()}

    tb = db.open_table(inverted_dict[book_type])
    records = tb.search(vector).limit(top_k).to_pandas()
    rag_str = ''
    for record_index, record in records.iterrows():
        rag_str += f'' \
                   f'相关文档{record_index}:\n' \
                   f'标题\提问:{record["title"]}\n' \
                   f'内容\回答:{record["content"]}\n' \
                   f'来源:{os.path.basename(record["file_from"])}的第{record["page_count"]}页/行\n\n'
    return rag_str


def local_chat(textbox, show_history, system_state, history, model_type, parm_b, steam_check_box, book_type):
    '''

    :param textbox: 用户提问
    :param show_history: 聊天组件
    :param system_state: 角色设定
    :param history: 历史记录
    :param model_type: 模型类别
    :param parm_b: 模型名字
    :param steam_check_box: 流式输出与否，str类型
    :param book_type: rag的名字
    :return:
    '''
    torch.cuda.empty_cache()
    if book_type == '不使用上下文':
        rag_str = '无'
    elif str(book_type) == 'None':
        rag_str = '无'
    else:
        rag_model, rag_tokenizer = load_rag_model('bge_m3')
        rag_str = add_rag_info(textbox, book_type, rag_model, rag_tokenizer, rag_database_name, rag_top_k)
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
        {'role':'user', 'content':f'相关文档：{rag_str},用户提问:{textbox}'}
    )
    model_name = model_type + '-' + parm_b
    model, tokenizer = load_model_cached(model_name)
    print(history)
    if len(steam_check_box) == 0:
        if model_name in qwen_support_list and steam_check_box==[]:
            response_message = qwen_model_detect(history, model, tokenizer)
        elif model_name == 'llama3-8b' and steam_check_box==[]:
            response_message = llama3_model_detect(history, model, tokenizer)
        elif model_name == 'MiniCPM3-4B' and steam_check_box == []:
            response_message = minicpm_model_detect(history, model, tokenizer)
        else:
            gr.Error(f'{model_name} not support!')
            assert False, 'model name not support!'
        response_dict = {'role': 'assistant', 'content': response_message}
        history.append(response_dict)
        show_history.append((textbox, response_message))
        yield '', show_history, history
    else:
        if model_name in qwen_support_list and len(steam_check_box)!=0 and steam_check_box[0]=='流式输出':
            conversion = tokenizer.apply_chat_template(history, add_generation_prompt=True, tokenize=False)
            model_inputs = tokenizer(conversion, return_tensors="pt").to('cuda')
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
        else:
            gr.Error(f'{model_name} not support')
            assert False, 'model name not support!'

if __name__ == '__main__':
    pass