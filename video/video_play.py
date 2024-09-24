import os
import random

import gradio as gr
import pandas as pd
from config import conf_yaml

root_dir = conf_yaml['video']['root_dir']
video_csv_path = conf_yaml['video']['mark_csv_path']

clothing_dict = conf_yaml['video']['clothing']
action_dict = conf_yaml['video']['action']
scene_dict = conf_yaml['video']['scene']
other_dict = conf_yaml['video']['other']
label_dict = {
    'clothing':clothing_dict,
    'action':action_dict,
    'scene':scene_dict,
    'other':other_dict
}

def get_csv_columns_name():
    csv2ch = {}
    ch2csv = {}
    init_list = [clothing_dict, action_dict, scene_dict, other_dict]
    for type_init_dict in init_list:
        for type_name, type_value in type_init_dict.items():
            csv2ch[type_name] = type_value
            ch2csv[type_value] = type_name
    columns_list = []
    columns_list.extend(['video_path', 'describe', 'start_score', 'breast_size'])
    columns_list.extend(list(csv2ch.keys()))
    return csv2ch, ch2csv, columns_list

csv2ch, ch2csv, columns_list = get_csv_columns_name()
def get_all_mp4_files(folder_paths):
    mp4_files = []
    for folder_path in folder_paths:
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.mp4'):
                    mp4_files.append(os.path.join(root, file))
    random.shuffle(mp4_files)
    return mp4_files


def get_random_mp4_file(folder_paths):
    mp4_files = get_all_mp4_files(folder_paths)

    if not mp4_files:
        return None  # 如果没有找到任何.mp4文件，返回None

    # 随机选择一个文件
    random_file = random.choice(mp4_files)
    return random_file

def init_dataframe(random_mp4_file):
    df = pd.DataFrame(columns=columns_list)
    info = {}
    info['video_path'] = random_mp4_file
    info['describe'] = ''
    info['start_score'] = ''
    info['breast_size'] = ''
    for csv_name in csv2ch.keys():
        info[csv_name] = False
    new_df = pd.DataFrame([info])
    df = pd.concat((df, new_df))
    clothing_boxs = []
    action_boxs = []
    scene_boxs = []
    other_boxs = []
    start_radio, breast_radio, describe_text = '', '', ''
    return random_mp4_file, random_mp4_file, start_radio, breast_radio, \
        clothing_boxs, action_boxs, scene_boxs, other_boxs, describe_text, df

def load_local_video():
    folder_paths = [root_dir]  # 替换为你的文件夹路径列表
    random_mp4_file = get_random_mp4_file(folder_paths)
    if os.path.exists(video_csv_path):
        df = pd.read_csv(video_csv_path)
        if len(df[df['video_path'] == random_mp4_file]) != 0:
            start_radio = df.loc[df['video_path'] == random_mp4_file, 'start_score'].values[0]
            breast_radio = df.loc[df['video_path'] == random_mp4_file, 'breast_size'].values[0]
            describle_str = str(df.loc[df['video_path'] == random_mp4_file, 'describe'] .values[0])
            describe_text = '' if describle_str=='nan' else describle_str
            clothing_boxs = []
            action_boxs = []
            scene_boxs = []
            other_boxs = []
            for csv_key, ch_value in csv2ch.items():
                type_flag = df.loc[df['video_path'] == random_mp4_file, csv_key].values[0]
                if type_flag == False:
                    continue
                else:
                    if csv_key in clothing_dict:
                        clothing_boxs.append(csv2ch[csv_key])
                        continue
                    if csv_key in action_dict:
                        action_boxs.append(csv2ch[csv_key])
                        continue
                    if csv_key in scene_dict:
                        scene_boxs.append(csv2ch[csv_key])
                        continue
                    if csv_key in other_dict:
                        other_boxs.append(csv2ch[csv_key])
                        continue
            return random_mp4_file, random_mp4_file, start_radio, breast_radio, \
                clothing_boxs, action_boxs, scene_boxs, other_boxs, describe_text
        else:
            random_mp4_file, random_mp4_file, start_radio, breast_radio, clothing_boxs, \
                action_boxs, scene_boxs, other_boxs, describe_text, new_df = init_dataframe(random_mp4_file)
            df = pd.concat((df, new_df))
            df.to_csv(video_csv_path, index=False)
            return random_mp4_file, random_mp4_file, start_radio, breast_radio, \
                clothing_boxs, action_boxs, scene_boxs, other_boxs, describe_text
    else:
        random_mp4_file, random_mp4_file, start_radio, breast_radio, clothing_boxs, \
            action_boxs, scene_boxs, other_boxs, describe_text, df = init_dataframe(random_mp4_file)
        df.to_csv(video_csv_path, index=False)
        return random_mp4_file, random_mp4_file, start_radio, breast_radio, \
            clothing_boxs, action_boxs, scene_boxs, other_boxs, describe_text


def mark_video_like(
        video_path,
        start_radio, breast_radio,
        clothing_boxs, action_boxs, scene_boxs, other_boxs,
        describe_text
):
    if os.path.exists(video_csv_path):
        df = pd.read_csv(video_csv_path)
    else:
        df = pd.DataFrame(columns=columns_list)
        df['video_path'] = video_path
    df.loc[df['video_path'] == video_path, 'start_score'] = start_radio
    df.loc[df['video_path'] == video_path, 'breast_size'] = breast_radio
    df.loc[df['video_path'] == video_path, 'describe'] = describe_text
    df.loc[df['video_path'] == video_path, list(csv2ch.keys())] = False
    value_list = [clothing_boxs, action_boxs, scene_boxs, other_boxs]
    for set_value_list in value_list:
        select_csv2ch = []
        for ch in set_value_list:
            select_csv2ch.append(ch2csv[ch])
        df.loc[df['video_path'] == video_path, select_csv2ch] = True
    df.to_csv(video_csv_path, index=False, encoding='utf8')
    gr.Info('success', duration=2)

if __name__ == '__main__':
    load_local_video()