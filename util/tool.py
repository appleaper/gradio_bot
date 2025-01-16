import os.path
import re
import pandas as pd


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

if __name__ == '__main__':
    csv_path = '/home/pandas/snap/code/RapidOcr/database_data/rag/data_csv/权力进化论.csv'
    book_path = './权力进化论.md'
    csv2markdown(csv_path, book_path)

    # file_path = '/home/pandas/snap/doc/1_非对称风险.md'  # 替换为你实际的 Markdown 文件路径