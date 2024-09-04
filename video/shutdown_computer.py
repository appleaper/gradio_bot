import subprocess

def shutdown_computer():
    # 定义要执行的脚本路径
    script_path = './shutdown_script.sh'

    # 使用subprocess.run执行脚本
    try:
        result = subprocess.run([script_path], check=True)
        print(f"脚本执行成功，返回码: {result.returncode}")
    except subprocess.CalledProcessError as e:
        print(f"脚本执行失败，错误信息: {e}")