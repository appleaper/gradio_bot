import torch
from transformers import AutoModel, AutoTokenizer
from utils.tool import singleton

@singleton
class OCR_AiGot():
    def __init__(self, device_str='cuda'):
        self.device_str = device_str
        self.model_name = ''
        self.platform = 'local'

    def check_device(self, device):
        if device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def init_mdoel(self, model_dir):
        '''初始化模型'''
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_dir, trust_remote_code=True, low_cpu_mem_usage=True,
            device_map=self.device_str, use_safetensors=True, pad_token_id=self.tokenizer.eos_token_id)
        self.model.eval().to()

    def parse_image(self, image_path):
        '''模型推理'''
        ocr_result = self.model.chat(self.tokenizer, image_path, ocr_type='format')
        return ocr_result

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

