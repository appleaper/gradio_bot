import os
import re
import pandas as pd
from utils.tool import read_md_doc
from utils.tool import hash_code, slice_string, chunk_str

def split_by_heading(text, level=2):
    # 使用正则表达式匹配二级标题
    pattern = r'(' +'#'*level +r' .+?)(?=\n##|\Z)'
    matches = re.findall(pattern, text, re.DOTALL)

    # 将匹配到的标题和内容分割成列表
    sections = []
    for match in matches:
        # 将标题和内容分开
        title, content = match.split('\n', 1)
        sections.append({'title': title.strip(), 'content': content.strip()})
    return sections

def parse_markdown_do(file_path, user_id, database_type, emb_model_class):
    markdown_data = read_md_doc(file_path)
    emb_model_name = emb_model_class.model_name
    markdown_data_list = split_by_heading(markdown_data, level=2)
    file_name, _ = os.path.splitext(os.path.basename(file_path))
    article_id = hash_code(file_name)
    result_list = []
    for info_i in markdown_data_list:
        text_list = slice_string(info_i, punctuation=r'[，。！？,.]')
        for content in text_list:
            info = chunk_str('', content, user_id, article_id, '', emb_model_name, database_type, file_name,
                             emb_model_class)
            result_list.append(info)
    result_df = pd.DataFrame(result_list)
    return result_df