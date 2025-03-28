import os
import shutil
import hashlib
import subprocess
import pandas as pd
from local.rag.parse.voice_parse import transcribe_audio
from local.rag.parse.fireredasr.models.fireredasr import FireRedAsr
from local.rag.rag_model import load_bge_model_cached

from utils.config_init import bge_m3_model_path, voice_model_path, voice_chunk_size, tmp_dir_path



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
            subprocess.run(extract_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
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


def parse_video_do(file_name, id, user_id, database_type):
    '''对音频进行解析'''
    if database_type in ['lancedb', 'milvus']:
        model_bge = load_bge_model_cached(bge_m3_model_path)
    model_voice = FireRedAsr.from_pretrained("aed", voice_model_path)
    info_list = []
    voice_result_list = extract_and_process_audio(file_name, tmp_dir_path, model_voice)
    voice_str = ''
    for result in voice_result_list:
        voice_str += result[0]['text']
    for i in range(0, len(voice_str), voice_chunk_size):
        chunk = voice_str[i:i + voice_chunk_size]
        info = {}
        info['user_id'] = user_id
        info['article_id'] = id
        info['title'] = ''
        info['content'] = chunk
        info['page_count'] = ''
        if database_type in ['lancedb', 'milvus']:
            info['vector'] = model_bge.encode(chunk, batch_size=1, max_length=8192)['dense_vecs'].tolist()
        else:
            info['vector'] = []
        info['file_from'] = os.path.basename(file_name)
        info['hash_check'] = hashlib.sha256((user_id+id+chunk).encode('utf-8')).hexdigest()
        info_list.append(info)
    df = pd.DataFrame(info_list)
    return df


if __name__ == '__main__':
    input_video = "/home/pandas/文档/d3eca89d47e62c61490970e0081679bb.mp4"
    df = parse_video_do(input_video)
    print(df.iloc[0])
    print(df.iloc[0].content)