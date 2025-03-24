from utils.config_init import name2path
from local.rag.parse.fireredasr.models.fireredasr import FireRedAsr

def load_voice_model(model_name):
    if model_name == 'FireRedAsr':
        model = FireRedAsr.from_pretrained("aed", name2path['FireRedAsr'])
        tokenizer = None
        return model, tokenizer
    else:
        assert False, 'voice model not support!'