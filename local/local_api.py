import torch
import hashlib
import functools
from local.qwen.qwen_api import qwen_model_init,qwen_model_detect
from local.llama3.llama3_api import llama3_model_init, llama3_model_detect
from config import conf_yaml
from database_data.decision_making import add_rag_info
from database_data.emb_model.init_model_dmeta import init_model as dmeta_model_init
from threading import Thread
from transformers import TextIteratorStreamer

name2path = conf_yaml['local_chat']['name2path']
max_history_len = conf_yaml['local_chat']['max_history_len']
def load_model(model_name):
    if model_name == 'qwen-1.5B':
        model, tokenizer = qwen_model_init(name2path['qwen-1.5B'])
    elif model_name == 'qwen-7B':
        model, tokenizer = qwen_model_init(name2path['qwen-7B'])
    elif model_name == 'llama3-8b':
        model, tokenizer = llama3_model_init(name2path['llama3-8b'])
    else:
        assert False, 'model name not support!'
    return model, tokenizer

def load_rag_model(model_name):
    if model_name == 'Dmeta-embedding-zh':
        model, tokenizer = dmeta_model_init(conf_yaml['rag']['decision']['model_path'])
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

def local_chat(textbox, show_history, system_state, history, model_type, parm_b, steam_check_box, book_type):
    torch.cuda.empty_cache()
    rag_model, rag_tokenizer = load_rag_model('Dmeta-embedding-zh')
    rag_textbox = add_rag_info(textbox, book_type, conf_yaml['rag']['max_rag_len'], rag_model, rag_tokenizer)
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
        {'role':'user', 'content':rag_textbox}
    )
    model_name = model_type + '-' + parm_b
    model, tokenizer = load_model_cached(model_name)
    print(history)
    if len(steam_check_box) == 0:
        if model_name == 'qwen-1.5B' and steam_check_box==[]:
            response_message = qwen_model_detect(history, model, tokenizer)
        elif model_name == 'qwen-7B' and steam_check_box==[]:
            response_message = qwen_model_detect(history, model, tokenizer)
        elif model_name == 'llama3-8b' and steam_check_box==[]:
            response_message = llama3_model_detect(history, model, tokenizer)
        else:
            assert False, 'model name not support!'
        response_dict = {'role': 'assistant', 'content': response_message}
        history.append(response_dict)
        show_history.append((textbox, response_message))
        yield '', show_history, history
    else:
        if model_name in ['qwen-1.5B', 'qwen-7B'] and len(steam_check_box)!=0 and steam_check_box[0]=='流式输出':
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
                    yield '', show_history,history
        else:
            assert False, 'model name not support!'



if __name__ == '__main__':
    pass