import numpy as np
from FlagEmbedding import BGEM3FlagModel

def model_init(model_path = '/home/pandas/snap/model/bge-small-zh-v1.5'):
    model = BGEM3FlagModel(model_path, use_fp16=True)
    return model, ''

def model_detect(sentences, model):
    sentence_embeddings = np.array(model.encode(sentences, batch_size=1, max_length=8192)['dense_vecs'])
    return sentence_embeddings
