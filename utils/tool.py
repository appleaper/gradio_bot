import os.path
import re
import pandas as pd
import pickle as pkl
import json

def load_html_file(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        html_content = file.read()
    return html_content

def add_hash(match):
    return '# ' + match.group(1)
def csv2markdown(csv_path, book_path):
    df = pd.read_csv(csv_path)
    out_str_list = []
    pattern = r'(第[零一二三四五六七八九十百千万]+章)'
    for index, row in df.iterrows():
        input_str =  row.content
        input_str = input_str.replace('\\title{', '')
        input_str = input_str.replace('\\', '')
        input_str = input_str.replace('section*{', '')
        input_str = input_str.replace('section{', '')
        input_str = input_str.replace('\section*{', '')
        input_str = input_str.replace('{^{', '')
        input_str = input_str.replace('{^{', '')
        input_str = input_str.replace('}', '')
        input_str = input_str.replace(' ', '')
        input_str = input_str.replace('begin{abstract', '')
        input_str = input_str.replace('end{abstract', '')
        input_str = input_str.replace('({^{oplus)', '')
        input_str = input_str.replace('(cdot)', '*')
        input_str = input_str.replace('author{', '')
        input_str = input_str.replace('sim', '~')
        input_str = input_str.replace('({^{', '')
        input_str = input_str.replace('({^{circledR)', '')
        input_str = input_str.replace('footnotetext{', '')
        input_str = input_str.replace('(cdot)', '*')
        input_str = input_str.replace('(mathrm{WER)', '')
        input_str = input_str.replace('mathrm{', '')
        input_str = input_str.replace('^{prime', '')
        input_str = input_str.replace('cdots', '……')
        input_str = input_str.replace('(mathrm{F)', '')
        input_str = input_str.replace('text{', '')
        input_str = input_str.replace('(Delta)', '')
        matches = re.findall(pattern, input_str)
        if len(matches) != 0:
            input_str = re.sub(pattern, add_hash, input_str)
        out_str_list.append(input_str)
    with open(book_path, 'w', encoding='utf8') as f:
        f.writelines(out_str_list)

def parse_md_file(file_path):
    result = []
    title = None
    content = []
    with open(file_path, 'r') as file:
        for line in file:
            stripped_line = line.strip()
            if stripped_line.startswith("##"):
                if title:
                    result.append({"title": title, "content": "\n".join(content)})
                title = stripped_line
                content = []
            else:
                content.append(line.strip())
    if title:
        result.append({"title": title, "content": "\n".join(content)})
    df = pd.DataFrame(result)
    filename, suffix = os.path.splitext(file_path)
    target_filename = os.path.join(filename+'.csv')
    df.to_csv(target_filename, index=False, encoding='utf8')
    return result

def save_data(data, file_path):
    """
    保存数据到指定文件
    :param data: 要保存的数据
    :param file_path: 保存文件的路径
    """
    try:
        # 以二进制写入模式打开文件
        with open(file_path, 'wb') as file:
            # 使用 pickle.dump 将数据序列化并写入文件
            pkl.dump(data, file)
        # print(f"数据已成功保存到 {file_path}")
    except Exception as e:
        print(f"保存数据时出错: {e}")

def load_data(file_path):
    """
    从指定文件读取数据
    :param file_path: 读取文件的路径
    :return: 读取的数据
    """
    try:
        # 以二进制读取模式打开文件
        with open(file_path, 'rb') as file:
            # 使用 pickle.load 从文件中反序列化数据
            data = pkl.load(file)
        return data
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到。")
        return None
    except Exception as e:
        print(f"读取数据时出错: {e}")
        return None

def read_json_file(file_path):
    """
    读取 JSON 文件并将其内容转换为字典
    :param file_path: JSON 文件的路径
    :return: 包含 JSON 数据的字典
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到。")
    except json.JSONDecodeError:
        print(f"无法解析 {file_path} 中的 JSON 数据。")
    return {}

def save_json_file(data, file_path):
    """
    将字典保存为 JSON 文件
    :param data: 要保存的字典数据
    :param file_path: 保存 JSON 文件的路径
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            # indent=4 用于美化输出，使 JSON 文件更易读
            json.dump(data, file, ensure_ascii=False, indent=4)
        #print(f"数据已成功保存到 {file_path}。")
    except Exception as e:
        print(f"保存数据到 {file_path} 时出现错误: {e}")

if __name__ == '__main__':
    csv_path = '/home/pandas/snap/code/RapidOcr/database_data/rag/data_csv/中国历代政治得失分块版.csv.csv'
    book_path = './中国历代政治得失分块版.md'
    csv2markdown(csv_path, book_path)

    # file_path = '/home/pandas/snap/doc/1_非对称风险.md'  # 替换为你实际的 Markdown 文件路径