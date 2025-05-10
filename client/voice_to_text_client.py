import json
import pandas as pd
import requests

global_port = '4500'
global_ip = '127.0.0.1'

def stand_alone_speech_client(speech_recognition_file, model_dir):
    data = {
        'speech_recognition_file': speech_recognition_file,
        'model_dir': model_dir,
    }
    response = requests.post(f'http://{global_ip}:{global_port}/stand_alone_speech', json=data)
    print(response)
    json_dict = response.json()
    speech_recognition_output_text = json_dict['speech_recognition_output_text']
    speech_recognition_file = json_dict['speech_recognition_file']
    error = json_dict['error']
    return speech_recognition_output_text, speech_recognition_file