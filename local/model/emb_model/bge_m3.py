import numpy as np
import torch
from FlagEmbedding import BGEM3FlagModel
from utils.tool import singleton
from local.model.emb_base_model import EmbBase

@singleton
class EmbBgeM3(EmbBase):
    def __init__(self, device='cuda'):
        self.check_device(device)
        self.platform = 'local'
        self.model_name = 'BAAI/bge-m3'


    def check_device(self, device):
        if device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def load_model(self, model_dir):
        # 加载模型时指定设备
        self.model = BGEM3FlagModel(model_dir, use_fp16=True, device=self.device)
        return self.model

    def parse_single_sentence(self, sentence, model_name=None):
        '''解析单个句子'''
        res = self.model.encode(sentence, batch_size=1, max_length=8192)['dense_vecs']
        sentence_embeddings = res.reshape(1,-1).tolist()
        return sentence_embeddings

    def unload_model(self):
        '''卸载模型'''
        if self.model is not None:
            try:
                # 将模型移到 CPU 以释放 GPU 显存
                self.model = self.model.to('cpu')
                # 释放未使用的缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.model = None
                return True
            except Exception:
                return False
        return True

if __name__ == '__main__':
    model_dir = r'C:\use\model\BAAIbge-m3'
    model_class = EmbBgeM3()
    model_class.load_model(model_dir)
    input_txt = '大海是什么颜色'
    vector = model_class.parse_single_sentence(input_txt)
    print(vector)
