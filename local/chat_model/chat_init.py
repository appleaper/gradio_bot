import gradio as gr
from local.qwen.qwen_api import qwen_model_init
from local.MiniCPM.minicpm_api import minicpm_model_init


def load_model(model_name):
    if model_name in qwen_support_list:
        model, tokenizer = qwen_model_init(name2path[model_name])
    elif model_name == 'llama3-8b':
        model, tokenizer = llama3_model_init(name2path['llama3-8b'])
    elif model_name == 'MiniCPM3-4B':
        model, tokenizer = minicpm_model_init(name2path['MiniCPM3-4B'])
    else:
        gr.Error(f'{model_name} not support!')
        assert False, f'model {model_name} name not support!'
    return model, tokenizer