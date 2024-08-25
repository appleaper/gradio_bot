import os
import random


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

root_dir = '/media/pandas/DataHaven1/video'
def load_local_video():
    folder_paths = [root_dir]  # 替换为你的文件夹路径列表
    random_mp4_file = get_random_mp4_file(folder_paths)

    return random_mp4_file

if __name__ == '__main__':
    load_local_video()