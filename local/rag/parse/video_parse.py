import os
import re
import uuid
import shutil
import hashlib
import subprocess

import gradio as gr
import pandas as pd
from local.rag.parse.voice_parse import transcribe_audio
from local.model.voice_model.fireredasr.models.fireredasr import FireRedAsr
from utils.tool import hash_code, slice_string, chunk_str
from local.model.voice_model.firereadasr_aed import VoiceAED



def extract_and_process_audio(input_video_path, output_audio_dir, model):
    all_results = []
    try:
        # 确保输出目录存在
        if not os.path.exists(output_audio_dir):
            os.makedirs(output_audio_dir)

        # 第一步：从视频中提取音频并重新编码为 MP3
        temp_audio_path = os.path.join(output_audio_dir, "temp_audio.mp3")
        extract_command = [
            'ffmpeg',
            '-y',  # 添加 -y 参数，自动覆盖已存在的文件
            '-i', input_video_path,
            '-vn',  # 不处理视频流
            '-acodec', 'libmp3lame',  # 将音频重新编码为 MP3
            temp_audio_path
        ]
        try:
            subprocess.run(extract_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, encoding='utf-8')
        except subprocess.CalledProcessError as e:
            print(f"执行提取音频命令时出现错误: {e.stderr.strip()}")
            return all_results

        # 第二步：对提取的音频进行处理
        processed_audio_path = os.path.join(output_audio_dir, "processed_audio.wav")
        process_command = [
            'ffmpeg',
            '-y',  # 添加 -y 参数，自动覆盖已存在的文件
            '-i', temp_audio_path,
            '-ar', '16000',  # 设置采样率为 16000Hz
            '-ac', '1',  # 设置声道数为 1（单声道）
            '-acodec', 'pcm_s16le',  # 设置音频编码为 PCM 16 位有符号小端序
            '-f', 'wav',  # 设置输出格式为 WAV
            processed_audio_path
        ]
        try:
            subprocess.run(process_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        except subprocess.CalledProcessError as e:
            print(f"执行处理音频命令时出现错误: {e.stderr.strip()}")
            return all_results

        # 第三步：获取音频时长
        duration_command = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            processed_audio_path
        ]
        result = subprocess.run(duration_command, capture_output=True, text=True)
        try:
            duration = float(result.stdout.strip())
        except ValueError:
            print("获取音频时长时出现错误，无法解析时长信息。")
            return all_results

        # 第四步：如果音频时长超过 60 秒，进行切割并推理
        if duration > 60:
            segment_num = 1
            start_time = 0
            while start_time < duration:
                end_time = start_time + 60
                if end_time > duration:
                    end_time = duration
                output_segment_path = os.path.join(output_audio_dir, f"segment_{segment_num}.wav")
                segment_command = [
                    'ffmpeg',
                    '-y',  # 添加 -y 参数，自动覆盖已存在的文件
                    '-i', processed_audio_path,
                    '-ss', str(start_time),
                    '-to', str(end_time),
                    '-c', 'copy',
                    output_segment_path
                ]
                try:
                    subprocess.run(segment_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
                except subprocess.CalledProcessError as e:
                    print(f"执行切割音频命令时出现错误: {e.stderr.strip()}")
                    continue

                # 对切割后的音频片段进行推理
                results = transcribe_audio(output_segment_path, model)
                all_results.append(results)
                # print(f"Segment {segment_num} 推理结果: {results}")

                start_time = end_time
                segment_num += 1
        else:
            # 音频时长不超过 60 秒，直接对处理后的音频进行推理
            output_path = os.path.join(output_audio_dir, "output.wav")
            shutil.copyfile(processed_audio_path, output_path)

            results = transcribe_audio(output_path, model)
            all_results.append(results)
            # print(f"完整音频推理结果: {results}")

        # 删除临时文件
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        if os.path.exists(processed_audio_path):
            os.remove(processed_audio_path)

        # print(f"音频已成功提取、处理、切割（如有需要）并完成推理，结果已打印。音频文件保存到 {output_audio_dir}")
    except FileNotFoundError:
        print("未找到 ffmpeg 或 ffprobe 可执行文件，请确保它们已正确安装并添加到系统 PATH 中。")
    return all_results

def contains_chinese(text):
    pattern = re.compile(r'[\u4e00-\u9fa5]')
    return bool(pattern.search(text))

def copy_file_with_chinese_name(source_path, temp_dir):
    try:
        if os.path.exists(source_path):
            if contains_chinese(os.path.dirname(source_path)):
                gr.Warning('路径中包含中文，请修改为全英文路径')
                return ''
            else:
                video_file_name = os.path.basename(source_path)
                if contains_chinese(video_file_name):
                    file_name, suffix = os.path.splitext(video_file_name)
                    unique_filename = str(uuid.uuid4().hex) + suffix
                    destination_path = os.path.join(temp_dir, unique_filename)
                    shutil.copy2(source_path, destination_path)
                    return destination_path
                else:
                    return source_path
        else:
            gr.Warning(f'文件不存在！')
            return ''
    except Exception as e:
        print(f"复制文件时出现错误: {e}")
        return ''

def parse_video_do(file_path, user_id, database_type, embedding_class, config_info):
    '''对音频进行解析'''

    voice_model = VoiceAED()
    '''对音频进行解析'''
    voice_model.init_mdoel(config_info['local_model_name_path_dict']['local_FireRedASR-AED-L'])
    tmp_dir = config_info['tmp_dir']
    file_name, _ = os.path.splitext(os.path.basename(file_path))
    file_path = copy_file_with_chinese_name(file_path, tmp_dir)
    if file_path == '':
        return pd.DataFrame([])
    else:
        voice_result_list = extract_and_process_audio(file_path, tmp_dir, voice_model.model)
        if os.path.exists(file_path):
            os.remove(file_path)

        voice_str = ''
        for result in voice_result_list:
            voice_str += result[0]['text']

        article_id = hash_code(file_name)
        emb_model_name = embedding_class.model_name
        info_list = []
        if len(voice_str) > 0:
            text_list = slice_string(voice_str, punctuation=r'[，。！？,.]')
            for text in text_list:
                info = chunk_str('', text, user_id, article_id, '', emb_model_name, database_type, file_name, embedding_class)
                info_list.append(info)
        df = pd.DataFrame(info_list)
        return df


if __name__ == '__main__':
    input_video = "/home/pandas/文档/d3eca89d47e62c61490970e0081679bb.mp4"
    df = parse_video_do(input_video)
    print(df.iloc[0])
    print(df.iloc[0].content)