import pandas as pd
from tqdm import tqdm
from database_data.emb_model.init_model_dmeta import init_model as dmeta_model_init
from database_data.emb_model.init_model_dmeta import model_detect as dmeta_model_detect


from database_data.emb_model.init_model_bge_small import model_init as bge_model_init
from database_data.emb_model.init_model_bge_small import model_detect as bge_model_detect

from database_data.lancedb_base_operations import create_table, connect_database, \
    add_data_to_table, open_exist_table, delete_table
from config import conf_yaml

def model_init(model_path, model_name):
    if model_name == 'dmeta':
        model, tokenizer = dmeta_model_init(model_path)
    elif model_name == 'bge_small':
        model, tokenizer = bge_model_init(model_path)
    else:
        assert False, f'{model_name} not support!'
    return model, tokenizer

def model_detect(text, model, tokenizer, model_name):
    if model_name == 'dmeta':
        emb = dmeta_model_detect(text, model, tokenizer)
    elif model_name == 'bge_small':
        emb = bge_model_detect(text, model, tokenizer)
    else:
        assert False, f'{model_name} not support!'
    return emb
def embbing_text(model_path, df, model_name):
    model, tokenizer = model_init(model_path, model_name)
    df.columns = [['title', 'content']]
    result_list = []
    for index, row in tqdm(df.iterrows(), total=len(df)):
        emb = model_detect(row.title + row.content, model, tokenizer, model_name)
        emb_list = emb.tolist()[0]
        info = {}
        info['vector'] = emb_list
        info['title'] = row.title
        info['content'] = row.content
        result_list.append(info)
    df = pd.DataFrame(result_list)
    return df

def create_database_table(model_path, data_path, database_name, table_name, model_name):
    df = pd.read_csv(data_path)
    db = connect_database(database_name)
    df = embbing_text(model_path, df, model_name)
    tb = open_exist_table(db, table_name)
    delete_table(db, table_name)
    # if tb.count_rows()>0:
    #     add_data_to_table(tb, df)
    # else:
    #     create_table(db, df, table_name)
    create_table(db, df, table_name)

def search_similar_info(database_name, table_name, model, tokenizer, content, top_k, model_name):
    db = connect_database(database_name)
    content_emb = model_detect(content, model, tokenizer, model_name).tolist()[0]
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
            model_name = conf_yaml['rag']['decision']['model_name']
            result = search_similar_info(database_name, table_name, model, tokenizer, textbox, top_k, model_name)
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



if __name__ == '__main__':
    # model_path = '/home/pandas/snap/model/Dmeta-embedding-zh'
    model_path = '/home/pandas/snap/model/bge-small-zh-v1.5'
    model_name = 'bge_small'        # bge_small, dmeta
    data_path = '/home/pandas/snap/code/RapidOcr/data/jue_ce_de_yi_shu/the_art_of_decision_making.csv'
    database_name = '/home/pandas/snap/code/RapidOcr/database_data/data/decision_making'
    table_name = 'decision'
    content = '我可以从这本书中获得什么？'
    top_k = 3
    model, tokenizer = model_init(model_path, model_name)
    create_database_table(model_path, data_path, database_name, table_name, model_name)
    result = search_similar_info(database_name, table_name, model, tokenizer, content, top_k, model_name)
    for i in result:
        print(i['title'])
        print(i['content'])