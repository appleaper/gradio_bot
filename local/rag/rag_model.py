
import hashlib
import functools
from FlagEmbedding import BGEM3FlagModel
from transformers import AutoModel, AutoTokenizer

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
@cached_model_loader
def load_model_cached(model_path):
    return load_model(model_path)

@cached_model_loader
def load_bge_model_cached(model_path):
    model_bge = BGEM3FlagModel(model_path, use_fp16=True)
    return model_bge

def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, low_cpu_mem_usage=True, device_map='cuda',
                                      use_safetensors=True, pad_token_id=tokenizer.eos_token_id)
    model = model.eval().cuda()
    return model, tokenizer