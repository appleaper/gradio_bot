import os
import random
import pandas as pd
from config import conf_yaml

root_dir = conf_yaml['video']['root_dir']
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
    # random_mp4_file = '/media/pandas/DataHaven1/video/luxu/好看/乱伦秀.mp4'
    return random_mp4_file, random_mp4_file

def mark_video_like(mark_score, high_heel, describe_text, video_path):
    csv_path = './video/video_mark.csv'
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame({'filename':[], 'score':[], 'high_heel':[], 'silk_stockings':[],'describe':[]})

    silk_stockings_flag = False
    high_heel_flag = False
    if len(high_heel)>0:
        for out_str in high_heel:
            if out_str == '高根':
                high_heel_flag = True
                continue
            if out_str == '丝袜':
                silk_stockings_flag = True
                continue
    if video_path in df['filename'].values:
        df.loc[df['filename'] == video_path, 'score'] = mark_score
        df.loc[df['filename'] == video_path, 'high_heel'] = high_heel_flag
        df.loc[df['filename'] == video_path, 'silk_stockings'] = silk_stockings_flag
        df.loc[df['filename'] == video_path, 'describe'] = describe_text
    else:
        new_row = pd.DataFrame([
            {
                'filename':video_path,
                'score':mark_score,
                'high_heel':high_heel_flag,
                'silk_stockings':silk_stockings_flag,
                'describe':describe_text}
        ])
        df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(csv_path, index=False, encoding='utf8')

if __name__ == '__main__':
    load_local_video()