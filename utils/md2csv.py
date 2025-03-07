import pandas as pd
def read_md_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到，请检查文件路径是否正确。")
    except Exception as e:
        print(f"读取文件时发生错误: {e}")

if __name__ == '__main__':
    # 调用函数读取文件，将文件路径替换为你自己的.md 文件的路径
    name = '权力进化论'
    file_path = f'./{name}.md'
    content = read_md_file(file_path)
    temp_list = content.split('#')
    line_info_list = []
    for line in temp_list:
        if len(line) == 0:
            continue
        info = {}
        line_temp_list = line.split('\n\n')
        info['title'] = line_temp_list[0]
        info['content'] = ''.join(line_temp_list[1:])
        line_info_list.append(info)
    df = pd.DataFrame(line_info_list)
    df.to_csv(f'./{name}_整理版.csv', index=False, encoding='utf8')
