import gradio as gr
import requests
import pandas as pd
global_port = '4500'
global_ip = '127.0.0.1'

def analyze_images_client(ocr_files, model_dir):
    data = {
        "file_path_list": ocr_files,
        "model_dir": model_dir
    }

    response = requests.post(f'http://{global_ip}:{global_port}/analyze_images', json=data)
    json_dict = response.json()
    error = json_dict['error']
    if error == '':
        ocr_files = json_dict['empty_list']
        ocr_image = json_dict['first_file_path']
        ocr_output_text = json_dict['ocr_str']
        ocr_download_file = json_dict['save_path']
        return ocr_files, ocr_output_text, ocr_download_file, ocr_image
    else:
        gr.Info(error)

def psa_analysis_client(psa_img_path, psa_model_path):
    data = {
        "psa_img_path": psa_img_path,
        "psa_model_path": psa_model_path
    }

    response = requests.post(f'http://{global_ip}:{global_port}/psa_analysis_server', json=data)
    json_dict = response.json()
    error = json_dict['error']
    if error == '':
        save_path = json_dict['save_path']
        return save_path
    else:
        gr.Info(error)

def imgs_cls_predict_client(cls_img_path, cls_model_type):
    data = {
        'cls_img_path': cls_img_path,
        'cls_model_type': cls_model_type
    }
    response = requests.post(f'http://{global_ip}:{global_port}/direction_judgment', json=data)
    json_dict = response.json()
    error = json_dict['error']
    if error == '':
        cls_result = json_dict['cls_result']
        cls_img_path = json_dict['cls_img_path']
        return cls_result
    else:
        gr.Info(error)

def text_detection_client(det_img_path, det_model_type):
    data = {
        'det_img_path': det_img_path,
        'det_model_type': det_model_type
    }
    response = requests.post(f'http://{global_ip}:{global_port}/text_detection', json=data)
    json_dict = response.json()
    error =  json_dict['error']
    if error =='':
        det_result = json_dict['det_result']
        return det_result
    else:
        gr.Info(error)

def text_recognition_client(rec_img_path, rec_model_type):
    data = {
        'rec_img_path': rec_img_path,
        'rec_model_type': rec_model_type
    }
    response = requests.post(f'http://{global_ip}:{global_port}/text_recognition', json=data)
    json_dict = response.json()
    error = json_dict['error']
    if error == '':
        rec_text_result = json_dict['rec_text_result']
        return rec_text_result
    else:
        gr.Info(error)

def small_model_text_recognition_client(sys_img_path, sys_model_type):
    data = {
        'sys_img_path': sys_img_path,
        'sys_model_type': sys_model_type
    }
    response = requests.post(f'http://{global_ip}:{global_port}/small_model_text_recognition', json=data)
    json_dict = response.json()
    error = json_dict['error']
    if error == '':
        sys_df_result = pd.read_json(json_dict['sys_df_result'])
        sys_img_result = json_dict['sys_img_result']
        return sys_df_result, sys_img_result
    else:
        gr.Info(error)