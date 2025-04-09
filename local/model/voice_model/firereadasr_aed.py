import torch
from utils.tool import singleton
from local.model.voice_model.fireredasr.models.fireredasr import FireRedAsr


@singleton
class VoiceAED():
    def __init__(self, device_str='cuda'):
        self.device_str = device_str
        self.model_name = 'FireRedTeam/FireRedASR-AED-L'
        self.platform = 'local'

    def check_device(self, device):
        if device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def init_mdoel(self, model_dir):
        '''初始化模型'''
        self.model = FireRedAsr.from_pretrained("aed", model_dir)

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

