import os.path

import hashlib
import functools
from local.embedding_model.embedding_init import load_rag_model
from local.chat_model.chat_init import load_model
from local.voice.voice_model_init import load_voice_model

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

@cached_model_loader
def load_voice_cached(model_name):
    return load_voice_model(model_name)

if __name__ == '__main__':
    pass