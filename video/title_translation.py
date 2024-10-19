import os
import re
import pandas as pd
from tqdm import tqdm
from config import conf_yaml
from local.qwen.qwen_api import qwen_model_init, qwen_model_detect

def contains_chinese_or_japanese(text):
    # 正则表达式匹配日文假名字符
    japanese_pattern = re.compile(r'[\u3040-\u309f]|[\u30a0-\u30ff]')

    # 检查文本是否包含中文或日文字符
    contains_japanese = bool(japanese_pattern.search(text))
    return contains_japanese

def new_translation_title():
    dir_path = '/media/pandas/video/video/video/video'
    model_path = conf_yaml['local_chat']['name2path']['qwen2.5-7B-Instruct-AWQ']
    model, tokenizer = qwen_model_init(model_path)
    info_list = []
    for file in tqdm(os.listdir(dir_path)):
        re_parent = contains_chinese_or_japanese(file)
        if re_parent == True:
            prom = file + '将以上内容从日文翻译成中文，直接给出翻译结果即可'
            messages = [
                {"role": "system", "content": '你是一个专业的日语翻译专家，专注成人影视行业'},
                {'role': 'user', 'content': prom}
            ]
            response = qwen_model_detect(messages, model, tokenizer)
            if '翻译为中文是：' in response:
                response = re.search(r'翻译为中文是：(.*)', response).group(1).strip()
            elif '翻译' in response:
                response = re.search(r'翻译(.*)', response).group(1).strip()
        else:
            response = file
        info = {}
        info['org'] = file
        info['response'] = response
        info_list.append(info)
    df = pd.DataFrame(info_list)
    df.to_csv('./translation.csv', index=False, encoding='utf8')
def get_translation_title():
    df = pd.read_csv(conf_yaml['video']['translation_title_csv_path'])
    name_dict = {}
    for index, row in df.iterrows():
        org_name = row['org']
        title_name = row['response']
        if org_name not in name_dict:
            name_dict[org_name] = title_name
    return name_dict

if __name__ == '__main__':
    pass