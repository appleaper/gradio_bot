import os.path
import time
import json
import threading
import gradio as gr
from utils.tool import read_json_file, save_json_file


class AuthManager:
    def __init__(self, json_file_path):
        self.json_file_path = json_file_path
        self.current_auth = self.load_auth_from_json()
        self.start_update_thread()

    def init_user_art_kb_json_files_do(self, json_file_path, user_info):
        json_info_dict = read_json_file(json_file_path)
        for user_name in user_info.keys():
            if user_name not in json_info_dict.keys():
                json_info_dict[user_name] = {}
        save_json_file(json_info_dict, json_file_path)

    def load_auth_from_json(self):
        user_info = read_json_file(self.json_file_path)
        return user_info


    def update_auth_periodically(self):
        interval = 24 * 60 * 60
        while True:
            new_auth = self.load_auth_from_json()
            if new_auth:
                self.current_auth = new_auth
            time.sleep(interval)

    def start_update_thread(self):
        update_thread = threading.Thread(target=self.update_auth_periodically)
        update_thread.daemon = True
        update_thread.start()

    def verify_auth(self, username, password):
        # 直接从字典中查找用户名对应的密码进行验证
        stored_password = self.current_auth.get(username)
        return stored_password is not None and stored_password == password

if __name__ == "__main__":
    import sys
    if sys.platform == 'win32':
        json_file_path = r'C:\use\code\RapidOcr_small\config\auth.json'
    else:
        json_file_path = ''
    auth_manager = AuthManager(json_file_path)
    def greet(name):
        return f"Hello, {name}!"

    if 'gr' in globals():
        demo = gr.Interface(fn=greet, inputs="text", outputs="text")
        demo.launch(server_name='0.0.0.0', auth=auth_manager.verify_auth)