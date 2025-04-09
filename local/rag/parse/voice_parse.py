import os
import sys
import subprocess
import pandas as pd
from tqdm import tqdm
from utils.tool import hash_code, slice_string, chunk_str
from local.model.voice_model.firereadasr_aed import VoiceAED

def transcribe_audio(audio_path, model):
    """
    该方法用于对指定音频文件进行转录
    :param audio_path: 音频文件的路径
    :return: 转录结果
    """
    batch_uttid = ["BAC009S0764W0121"]
    batch_wav_path = [audio_path]

    # FireRedASR-AED

    results = model.transcribe(
        batch_uttid,
        batch_wav_path,
        {
            "use_gpu": 1,
            "beam_size": 3,
            "nbest": 1,
            "decode_max_len": 0,
            "softmax_smoothing": 1.0,
            "aed_length_penalty": 0.0,
            "eos_penalty": 1.0
        }
    )
    return results

def split_single_mp3_and_convert(input_file, output_folder, start_time, end_time, segment_index):
    """
    切割单个 MP3 片段并转换为指定格式的 WAV 文件
    :param input_file: 输入的 MP3 文件路径
    :param output_folder: 输出文件的文件夹路径
    :param start_time: 片段的起始时间
    :param end_time: 片段的结束时间
    :param segment_index: 片段的索引
    :return: 切割后的音频文件路径
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_file = os.path.join(output_folder, f"segment_{segment_index}.wav")

    # 检查文件是否存在，如果存在则删除
    if os.path.exists(output_file):
        try:
            os.remove(output_file)
        except OSError as e:
            print(f"删除文件 {output_file} 时出错: {e}")

    # 构建 ffmpeg 命令
    ffmpeg_cmd = [
        'ffmpeg',
        '-i', input_file,
        '-ss', str(start_time),
        '-to', str(end_time),
        '-ar', '16000',
        '-ac', '1',
        '-acodec', 'pcm_s16le',
        '-f', 'wav',
        output_file
    ]

    try:
        # 执行 ffmpeg 命令，抑制输出
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return output_file
    except subprocess.CalledProcessError as e:
        print(f"生成片段 {segment_index} 时出错: {e}")
        return None

def get_audio_duration(file_path):
    try:
        command = ['ffmpeg', '-i', file_path]
        # 明确指定 encoding 为 'utf-8'
        result = subprocess.run(command, stderr=subprocess.PIPE, text=True, encoding='utf-8')
        output = result.stderr
        duration_line = next((line for line in output.splitlines() if 'Duration' in line), None)
        if duration_line:
            duration_str = duration_line.split(' ')[3].rstrip(',')
            h, m, s = map(float, duration_str.split(':'))
            duration = h * 3600 + m * 60 + s
            return duration
        else:
            print("未找到时长信息")
            return None
    except Exception as e:
        print(f"发生错误: {e}")
        return None

def process_audio(input_mp3_file, output_folder, model):
    """
    处理音频文件，包括切割、转录和删除临时文件
    :param input_mp3_file: 输入的 MP3 文件路径
    :param output_folder: 输出文件的文件夹路径
    :param model_path: 模型的路径
    :return: 所有片段的转录结果
    """
    # 获取输入文件的时长（秒）
    if sys.platform == 'win32':
        total_duration = get_audio_duration(input_mp3_file)
    else:
        cmd = f'ffmpeg -i "{input_mp3_file}" -f null - 2>&1 | grep "Duration" | cut -d " " -f 4 | sed s/,// | sed s/\\\\./,/ | awk -F: \'{{ print ($1 * 3600) + ($2 * 60) + $3 }}\''
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        total_duration = float(result.stdout.strip())

    segment_duration = 60
    # 计算片段数量
    num_segments = int(total_duration // segment_duration) + (1 if total_duration % segment_duration != 0 else 0)

    voice_str = ''
    for i in tqdm(range(num_segments)):
        start_time = i * segment_duration
        end_time = min((i + 1) * segment_duration, total_duration)

        # 切割单个片段
        output_file = split_single_mp3_and_convert(input_mp3_file, output_folder, start_time, end_time, i)

        if output_file:
            # 对该片段进行转录
            result = transcribe_audio(output_file, model)
            voice_str += result[0]['text'] + '\n'

            # 删除临时切割的音频文件
            try:
                os.remove(output_file)
            except OSError as e:
                print(f"删除文件 {output_file} 时出错: {e}")

    return voice_str

voice_model = VoiceAED()
def parse_voice_do(file_path, user_id, database_type, embedding_class, config_info):
    '''对音频进行解析'''
    voice_model.init_mdoel(config_info['local_model_name_path_dict']['local_FireRedASR-AED-L'])
    tmp_dir = config_info['tmp_dir']
    file_name, _ = os.path.splitext(os.path.basename(file_path))
    voice_str = process_audio(file_path, tmp_dir, voice_model.model)
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


def standalone_voice_analysis(file_path, model_dir):
    '''独立的解析图片'''
    voice_model.init_mdoel(model_dir)
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    tmp_dir = os.path.join(project_dir, 'data', 'temp')
    voice_str = process_audio(file_path, tmp_dir, voice_model.model)
    return voice_str, []


if __name__ == '__main__':
    pass