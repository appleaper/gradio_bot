import lancedb

def connect_database(database_name):
    db = lancedb.connect(database_name)
    return db

def create_table(db, df, table_name):
    tb = db.create_table(table_name, data=df, exist_ok=True)
    return tb

def open_exist_table(db, table_name):
    tb = db.open_table(table_name)
    return tb

def show_table_name(db):
    return db.table_names()

def delete_table(db, table_name):
    db.drop_table(table_name)

def add_data_to_table(tb, df):
    tb.add(df)

def add_data_to_table_yield(tb, df):
    def make_batches():
        for i in range(5):
            yield [
                {"vector": [3.1, 4.1], "item": "peach", "price": 6.0},
                {"vector": [5.9, 26.5], "item": "pear", "price": 5.0}
            ]

    tb.add(make_batches())

def update_data_to_table(tb,condition,update_value):
    tb.update(where=condition, values=update_value)

def delete_data_to_table(tb, key, value):
    tb.delete(f'{key} = {value}')

def search_data_from_table(input_emb, top_k, tb):
    return tb.search(input_emb).limit(top_k).to_pandas()