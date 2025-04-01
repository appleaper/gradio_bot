import os.path
import subprocess
import sys
import gradio as gr

def shutdown_computer():
    # 定义要执行的脚本路径
    script_path = os.path.join(project_dir, 'utils', 'script', 'shutdown_script')


    # 使用subprocess.run执行脚本
    try:
        if sys.platform == 'win32':
            gr.Warning('window 系统暂时还不能执行关机命令')
        else:
            result = subprocess.run([script_path], check=True)
            gr.Info(f"脚本执行成功，返回码: {result.returncode}")
    except subprocess.CalledProcessError as e:
        print(f"脚本执行失败，错误信息: {e}")