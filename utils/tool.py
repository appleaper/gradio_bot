import os
import re
import uuid
import json
import ollama
import hashlib
import pandas as pd
import pickle as pkl
import yaml

def singleton(cls):
    instances = {}
    def wrapper(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return wrapper

def generate_unique_filename(extension='jpg'):
    unique_filename = str(uuid.uuid4()) + '.' + extension
    return unique_filename

def load_html_file(filepath):
    '''读取htmml文件'''
    with open(filepath, "r", encoding="utf-8") as file:
        html_content = file.read()
    return html_content

def read_yaml(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
            return data
    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 未找到。")
    except yaml.YAMLError as e:
        print(f"错误: 解析 YAML 文件时出错: {e}")
    except Exception as e:
        print(f"错误: 发生未知错误: {e}")
    return None

def add_hash(match):
    '''添加hash'''
    return '# ' + match.group(1)

def csv2markdown(csv_path, book_path):
    '''csv转markdown格式'''
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
    '''解析markdown文件'''
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

def encrypt_username(username):
    # 对用户名进行加密
    hash_object = hashlib.sha256(username.encode('utf-8'))
    hex_dig = hash_object.hexdigest()
    # 截取前 8 位作为加密后的用户名
    encrypted_username = hex_dig[:8]
    return encrypted_username

def save_rag_group_name(user_name, articles_name, articles_user_mapping_dict, mapping_json_path):
    '''给组名返回一个唯一的id'''
    user_info = articles_user_mapping_dict[user_name]
    if articles_name not in user_info.values():
        id = str(uuid.uuid4())[:8]
        user_info[id] = articles_name
    else:
        id_list = [key for key, value in user_info.items() if value == articles_name]
        id = id_list[0]
    articles_user_mapping_dict[user_name] = user_info
    save_json_file(articles_user_mapping_dict, mapping_json_path)
    return id

def reverse_dict(data):
    '''id和文章名字相反'''
    inverted_dict = {value: key for key, value in data.items()}
    return inverted_dict

def read_user_info_dict(user_name, path):
    with open(path, 'r', encoding='utf8') as json_file:
        json_dict = json.load(json_file)
    return json_dict[user_name]

def read_md_doc(path):
    with open(path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

def get_ollama_model_list():
    res = ollama.list().models
    model_list = []
    for model in res:
        model_list.append(model.model)
    return model_list

def generate_unique_id():
    '''生成唯一数'''
    unique_id = uuid.uuid4()
    unique_id_str = str(unique_id)
    unique_id_without_hyphen = unique_id_str.replace("-", "")
    return unique_id_without_hyphen

def hash_code(text):
    return hashlib.sha256((text).encode('utf-8')).hexdigest()

def slice_string(text, chunk_size=3000, punctuation=r'[。！？]'):
    '''对输入数据进行分块'''
    result = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        # 从 chunk_size 字处往前找标点符号
        while end > start and not re.match(punctuation, text[end - 1]):
            end -= 1
        # 如果没找到合适的标点，就按 chunk_size 字切
        if end == start:
            end = min(start + chunk_size, len(text))
        result.append(text[start:end])
        start = end
    return result

def chunk_str(title, content, user_id, article_id, index, emb_model_name, database_type, file_name, embedding_class=None):
    info = {}
    input_str = f'{title}\n{content}\n'
    info['user_id'] = user_id
    info['article_id'] = article_id
    info['title'] = title
    info['content'] = content
    if index == '' or index == None:
        info['page_count'] = ''
    else:
        info['page_count'] = str(index + 1)
    info['embed_model_name'] = emb_model_name
    info['platform'] = embedding_class.platform
    if database_type in ['lancedb', 'milvus']:
        vector = embedding_class.parse_single_sentence(
            model_name=emb_model_name,
            sentence=input_str
        )[0]
        info['vector'] = vector
    else:
        info['vector'] = []
    info['file_from'] = file_name
    info['database_type'] = database_type
    hash_content = user_id + article_id + input_str + emb_model_name + database_type
    info['hash_check'] = hashlib.sha256((hash_content).encode('utf-8')).hexdigest()
    return info

if __name__ == '__main__':
    csv_path = '/home/pandas/snap/code/RapidOcr/database_data/rag/data_csv/中国历代政治得失分块版.csv.csv'
    book_path = './中国历代政治得失分块版.md'
    csv2markdown(csv_path, book_path)

    # file_path = '/home/pandas/snap/doc/1_非对称风险.md'  # 替换为你实际的 Markdown 文件路径