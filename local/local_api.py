import hashlib
import functools
from local.qwen.qwen_api import qwen_model_init,qwen_model_detect
from local.llama3.llama3_api import llama3_model_init, llama3_model_detect

name2path = {
    'qwen-1.5B':'/home/pandas/snap/model/QwenQwen2-1.5B-Instruct',
    'qwen-7B':'/home/pandas/snap/model/QwenQwen2-7B-Instruct-AWQ',
    'llama3-8b':'/home/pandas/snap/model/Llama3-8B-Chinese-Chat',
}

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
def local_chat(textbox, show_history, system_state, history, model_type, parm_b, steam_check_box):
    if show_history is None:
        history = []
    if len(history) == 0:
        history = [{"role":"system","content":system_state}]
    elif len(history) >= 6:
        history = history[-6:]
        history[0] = {"role": "system", "content": system_state}
    else:
        history[0] = {"role":"system","content":system_state}
    history.append(
        {'role':'user', 'content':textbox}
    )
    model_name = model_type + '-' + parm_b
    model, tokenizer = load_model_cached(model_name)

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
            from threading import Thread
            from transformers import TextIteratorStreamer
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