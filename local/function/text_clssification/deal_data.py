import pandas as pd

if __name__ == '__main__':
    cls_list = ['finance''realty','stocks','education','science','society','politics','sports','game','entertainment']
    from utils.tool import read_text_file
    file_path = r'C:\use\code\Bert-Chinese-Text-Classification-Pytorch-master\THUCNews\data\test.txt'
    text_lines = read_text_file(file_path)
    info_list = []
    for line in text_lines:
        info = {}
        title, content = line.split('\t')
        info['content'] = title
        info['cls'] = cls_list[int(content)-1]
        info_list.append(info)
    df = pd.DataFrame(info_list)
    df.to_csv(r'C:\Users\APP\Desktop\gradio_bot\test.csv', index=False, encoding='utf-8')