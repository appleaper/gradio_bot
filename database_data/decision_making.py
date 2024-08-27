import pandas as pd
from tqdm import tqdm
from database_data.emb_model.init_model import model_detect,init_model
from learning.chromadb_learn.lancedb_base_operations import create_table, connect_database
from config import conf_yaml
def embbing_text(model_path, df):
    model, tokenizer = init_model(model_path)
    df.columns = [['title', 'content']]
    result_list = []
    for index, row in tqdm(df.iterrows(), total=len(df)):
        emb = model_detect(row.content, model, tokenizer)
        emb_list = emb.tolist()[0]
        info = {}
        info['vector'] = emb_list
        info['title'] = row.title
        info['content'] = row.content
        result_list.append(info)
    df = pd.DataFrame(result_list)
    return df

def create_database_table(model_path, data_path, database_name, table_name):
    df = pd.read_csv(data_path)
    db = connect_database(database_name)
    df = embbing_text(model_path, df)
    create_table(db, df, table_name)

def search_similar_info(database_name, table_name, model, tokenizer, content, top_k):
    db = connect_database(database_name)
    content_emb = model_detect(content, model, tokenizer).tolist()[0]
    table = db.open_table(table_name)
    result = table.search(content_emb).limit(top_k).to_list()
    return result

def add_rag_info(textbox, book_type, max_len, model, tokenizer):
    if book_type == None:
        return textbox
    else:
        if book_type == '决策的艺术':
            database_name = conf_yaml['rag']['decision']['database_name']
            table_name = conf_yaml['rag']['decision']['table_name']
            top_k = conf_yaml['rag']['decision']['top_k']
            result = search_similar_info(database_name, table_name, model, tokenizer, textbox, top_k)
            rag_str = ''
            prom = '你可以参考以上内容，来回答接下来的问题'
            use_len = max_len - len(textbox) - len(prom)
            for i in result:
                if len(i['title']) + len(i['content']) > use_len:
                    merge_str = i['title'] + i['content']
                    rag_str += merge_str[:use_len]
                    break
                else:
                    rag_str += i['title'] + i['content']
                    use_len -= len(i['title']) - len(i['content'])
            out_str = rag_str + prom + textbox
            return out_str
        else:
            assert False, f'{book_type} not support!'



# if __name__ == '__main__':
#     model_path = '/home/pandas/snap/model/Dmeta-embedding-zh'
#     data_path = '/home/pandas/snap/code/RapidOcr/learning/phone/data/决策的艺术.csv'
#     database_name = '/home/pandas/snap/code/RapidOcr/database_data/data/decision_making'
#     table_name = 'decision'
#     content = '产生更好备选方案的关键是什么？'
#     top_k = 3
#     model, tokenizer = init_model(model_path)
#     result = search_similar_info(database_name, table_name, model, tokenizer, content, top_k)
#     for i in result:
#         print(i['title'])
#         print(i['content'])