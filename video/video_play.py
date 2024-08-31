import os
import random

import gradio as gr
import pandas as pd
from config import conf_yaml

root_dir = conf_yaml['video']['root_dir']
video_csv_path = conf_yaml['video']['mark_csv_path']
csv2chinese_str_dict = conf_yaml['video']['hobby_tag']
def get_all_mp4_files(folder_paths):
    mp4_files = []
    for folder_path in folder_paths:
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.mp4'):
                    mp4_files.append(os.path.join(root, file))
    return mp4_files


def get_random_mp4_file(folder_paths):
    mp4_files = get_all_mp4_files(folder_paths)

    if not mp4_files:
        return None  # 如果没有找到任何.mp4文件，返回None

    # 随机选择一个文件
    random_file = random.choice(mp4_files)
    return random_file
def load_local_video():
    folder_paths = [root_dir]  # 替换为你的文件夹路径列表
    random_mp4_file = get_random_mp4_file(folder_paths)
    df = pd.read_csv(video_csv_path)
    select_df = df[df['filename'] == random_mp4_file]
    if len(select_df) !=0:
        start_radio = select_df.iloc[0]['score']
        type_check_boxs = []
        bool_columns = select_df[list(csv2chinese_str_dict.keys())]
        true_columns = bool_columns.columns[bool_columns.iloc[0] == True].tolist()
        for name in true_columns:
            type_check_boxs.append(csv2chinese_str_dict[name])
        describe = select_df.iloc[0]['describe']
    else:
        start_radio, type_check_boxs, describe = '', [], ''
    return random_mp4_file, random_mp4_file, start_radio, type_check_boxs, describe

def mark_video_like(mark_score, key_words, describe_text, video_path):
    if os.path.exists(video_csv_path):
        df = pd.read_csv(video_csv_path)
    else:
        info_dict = {}
        info_dict['filename'] = []
        info_dict['score'] = []
        info_dict['describe'] = []
        for key_name in csv2chinese_str_dict.keys():
            info_dict[key_name] = []
        df = pd.DataFrame(info_dict)

    chinese2csv_dict = {value: key for key, value in csv2chinese_str_dict.items()}
    select_df = df[df['filename'] == video_path]
    if len(select_df) > 0:
        row_index = select_df.index[0]
        columns_list = select_df.columns.values.tolist()
        columns_list.remove('filename')
        columns_list.remove('score')
        columns_list.remove('describe')
        for select_str in key_words:
            csv_str = chinese2csv_dict[select_str]
            df.loc[row_index, csv_str] = True
            columns_list.remove(chinese2csv_dict[select_str])
        df.loc[row_index, columns_list] = False
        df.loc[row_index, 'score'] = mark_score
        df.loc[row_index, 'describe'] = describe_text
    else:
        info_dict = {}
        info_dict['filename'] = [video_path]
        info_dict['score'] = [mark_score]
        info_dict['describe'] = [describe_text]
        for key_name in csv2chinese_str_dict.keys():
            info_dict[key_name] = [False]
        for select_str in key_words:
            csv_str = chinese2csv_dict[select_str]
            info_dict[csv_str] = [True]
        new_df = pd.DataFrame(info_dict)
        df = pd.concat((df, new_df))
    df.to_csv(video_csv_path, index=False, encoding='utf8')
    gr.Info('success', duration=2)

if __name__ == '__main__':
    load_local_video()