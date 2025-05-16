import json
import requests
import gradio as gr
import pandas as pd

global_port = '4500'
global_ip = '127.0.0.1'

def text_cls_train_client(data_path, model_dir, batch_size, num_epochs, save_dir):
    data = {
        'data_path': data_path,
        'model_dir': model_dir,
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'save_dir': save_dir
    }
    response = requests.post(f'http://{global_ip}:{global_port}/text_cls_train', json=data)
    json_dict = response.json()
    error = json_dict['error']
    if error == '':
        train_info_json = json_dict['train_info']
        train_info = pd.DataFrame(json.loads(train_info_json))
        return train_info
    else:
        gr.Info(error)

def text_cls_predict_client(model_dir, save_dir, pre_input_text):
    data = {
        'model_dir': model_dir,
        'save_dir': save_dir,
        'pre_input_text': pre_input_text,
    }
    response = requests.post(f'http://{global_ip}:{global_port}/text_cls_predict', json=data)
    json_dict = response.json()
    error = json_dict['error']
    if error:
        pre_result_json = json_dict['pre_result']
        pre_info = pd.DataFrame(json.loads(pre_result_json))
        return pre_info
    else:
        gr.Info(error)