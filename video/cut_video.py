import os.path
import subprocess
import gradio as gr
from config import conf_yaml

video_save_dir = conf_yaml['video']['cut_save_dir']
video_cut_record_path = conf_yaml['video']['cut_record_path']
def time_to_seconds(time_string):
    hours, minutes, seconds = map(int, time_string.split(':'))
    return hours * 3600 + minutes * 60 + seconds

def ffmpeg_video_cut(input_path, output_path, start_time_seconds, end_time_seconds):
    command = [
        "ffmpeg",
        "-i", input_path,
        "-ss", str(start_time_seconds),
        "-t", str(end_time_seconds - start_time_seconds),
        "-c", "copy",
        output_path
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the ffmpeg command: {e}")

def get_time_str(input_str):
    if int(input_str) < 10:
        input_str = '0' + str(input_str)
    else:
        input_str = str(input_str)
    return input_str
def video_cut(input_video_path, s_h, s_m, s_s, e_h, e_m, e_s):
    s_h = get_time_str(s_h)
    s_m = get_time_str(s_m)
    s_s = get_time_str(s_s)
    e_h = get_time_str(e_h)
    e_m = get_time_str(e_m)
    e_s = get_time_str(e_s)
    file_name = os.path.splitext(os.path.basename(input_video_path))[0]
    save_path = os.path.join(video_save_dir, file_name + f'{s_h}_{s_m}__to__{e_h}_{e_m}' +'.mp4')
    if os.path.exists(save_path):
        gr.Warning('video has been exist!')
        return '','','','','',''
    else:
        start_time_seconds = time_to_seconds(f'{s_h}:{s_m}:{s_s}')
        end_time_seconds = time_to_seconds(f'{e_h}:{e_m}:{e_s}')
        write_str = ''
        try:
            ffmpeg_video_cut(input_video_path, save_path, start_time_seconds, end_time_seconds)
            write_str += 'successful' + '\t'
            gr.Info('cut successful!')
            write_flag = True
        except:
            write_str += 'faild' + '\t'
            gr.Error('cut video error')
            write_flag = False

        if write_flag:
            with open(video_cut_record_path, 'a', encoding='utf8') as f:
                write_str += input_video_path + '\t' + save_path + '\n'
                f.writelines(write_str)