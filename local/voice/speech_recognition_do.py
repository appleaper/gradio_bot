import os.path
import gradio as gr
from local.qwen.qwen_api import qwen_model_detect
from local.rag.parse.voice_parse import process_audio
from local.local_api import load_voice_cached, load_model_cached

def parse_voice(file_name):
    voice_model, _ = load_voice_cached('FireRedAsr')
    voice_str = process_audio(file_name, tmp_dir_path, voice_model)
    return voice_str, []

def speech_recognition(file_name):
    basename, suffix = os.path.splitext(file_name)
    if suffix in ['.mp3', '.war']:
        return parse_voice(file_name)
    else:
        gr.Warning(f'not support {suffix} file')