import pandas as pd
from tqdm import tqdm
from config import conf_yaml
from local.qwen.qwen_api import qwen_model_init, qwen_model_detect

if __name__ == '__main__':
    csv_path = '/home/pandas/snap/code/RapidOcr/database_data/rag/data_csv/power_and_progress.csv'
    df = pd.read_csv(csv_path, encoding='utf8')
    dir_path = '/media/pandas/video/video/video/video'
    model_path = conf_yaml['local_chat']['name2path']['qwen2.5-7B-Instruct-AWQ']
    model, tokenizer = qwen_model_init(model_path)
    info_list = []
    for index, row in tqdm(df.iterrows(), total=len(df)):
        info = {}
        info['title'] = row.title
        info['content'] = row.content
        info['page_count'] = row.page_count
        info['vector'] = row.vector
        info['file_from'] = row.file_from
        messages = [
            {"role": "system", "content": '你是一个专业的英语翻译专家'},
            {'role': 'user', 'content': f'将以下内容翻译成中文:\n{row.content}'}
        ]
        response = qwen_model_detect(messages, model, tokenizer)
        info['response'] = response
        info_list.append(info)
    df = pd.DataFrame(info_list)
    df.to_csv('./power_and_progress.csv', index=False, encoding='utf8')