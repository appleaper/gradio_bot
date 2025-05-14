import os
import time
import torch
import numpy as np
import gradio as gr
import pandas as pd
import torch.nn as nn
from tqdm import tqdm

from sklearn import metrics
from datetime import timedelta
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils.tool import save_json_file, read_json_file
from sklearn.model_selection import train_test_split
from local.function.text_clssification.pytorch_pretrained.bert import BertModel, BertTokenizer, BertAdam
from flask import request, jsonify, Blueprint

PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号

class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        return (x, seq_len, mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches

class Model(nn.Module):
    def __init__(self, bert_path, hidden_size, num_classes):
        super(Model, self).__init__()
        self.bert_path = bert_path
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.bert = BertModel.from_pretrained(self.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out = self.fc(pooled)
        return out

class TextClassBert():
    def __init__(self, bert_path, batch_size, num_epochs, save_dir):
        self.bert_path = bert_path
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.batch_size = batch_size
        self.learning_rate = 5e-5
        self.num_epochs = int(num_epochs)
        self.model_name = 'bert'
        self.require_improvement = 15
        self.save_dir = save_dir
        self.pad_size = 30
        self.save_path = os.path.join(save_dir, self.model_name + '.ckpt')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hidden_size = 768
        self.cls2id_path = os.path.join(save_dir, 'cls2id.json')


    def get_cls2id(self, cls_list):
        cls2id = {}
        id2cls = {}
        for index, name in enumerate(cls_list):
            cls2id[name] = index
            id2cls[index] = name
        save_json_file(cls2id, self.cls2id_path)
        return cls2id, id2cls

    def split_dataframe(self, df, traom_rate=0.7, val_rate=0.2, test_rate=0.1):
        # 打乱 DataFrame 的顺序
        df = df.sample(frac=1, random_state=42)

        # 先拆分出训练集和剩余部分
        train_df, temp_df = train_test_split(df, train_size=traom_rate, random_state=42)

        # 再将剩余部分按照 2:1 的比例拆分成验证集和测试集
        val_df, test_df = train_test_split(
            temp_df, test_size=test_rate / (val_rate + test_rate), random_state=42)

        return train_df, val_df, test_df

    def process_csv_data(self, df, pad_size=512):
        contents = []
        for index, row in tqdm(df.iterrows(), total=len(df)):
            ask = row['content']
            if str(ask) == 'nan':
                continue

            if str(row['cls']) == 'nan':
                continue

            label = self.cls2id[row['cls']]
            token = self.tokenizer.tokenize(ask)
            token = [CLS] + token
            seq_len = len(token)
            mask = []
            token_ids = self.tokenizer.convert_tokens_to_ids(token)
            if pad_size:
                if len(token) < pad_size:
                    mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                    token_ids += ([0] * (pad_size - len(token)))
                else:
                    mask = [1] * pad_size
                    token_ids = token_ids[:pad_size]
                    seq_len = pad_size
            contents.append((token_ids, int(label), seq_len, mask))
        return contents

    def plot_dataframe_stats(self, df, save_dir):
        # 设置字体以支持中文显示
        plt.rcParams['font.sans-serif'] = ['SimHei']
        # 解决负号显示问题
        plt.rcParams['axes.unicode_minus'] = False

        # 统计每一行 content 的字符串长度
        df['content_length'] = df['content'].apply(len)

        # 绘制字符长度的直方图并保存
        plt.figure()
        plt.hist(df['content_length'], bins=5, edgecolor='black')
        plt.xlabel('字符长度')
        plt.ylabel('频数')
        plt.title('content 字符长度直方图')
        plt.savefig(os.path.join(save_dir, 'content_length_histogram.png'))
        plt.close()

        # 统计 cls 列每个类的频数
        cls_counts = df['cls'].value_counts()

        # 绘制饼状图并保存
        plt.figure()
        plt.pie(cls_counts, labels=cls_counts.index, autopct='%1.1f%%')
        plt.title('cls 列各类频数饼状图')
        plt.axis('equal')
        plt.savefig(os.path.join(save_dir, 'cls_counts_pie_chart.png'))
        plt.close()

    def get_train_val_test_data(self, path):
        df = pd.read_csv(path, encoding='utf-8')
        df = df.drop_duplicates()
        df = df.dropna()
        self.plot_dataframe_stats(df, self.save_dir)
        class_name_list = list(df['cls'].unique())
        self.cls2id, self.id2cls = self.get_cls2id(class_name_list)
        train_data, val_data, test_data = self.split_dataframe(df)
        train_ = self.process_csv_data(train_data, self.pad_size)
        val_ = self.process_csv_data(val_data, self.pad_size)
        test_ = self.process_csv_data(test_data, self.pad_size)
        return train_, val_, test_


    def build_iterator(self, dataset):
        iter = DatasetIterater(dataset, self.batch_size, self.device)
        return iter

    def get_time_dif(self, start_time):
        """获取已使用时间"""
        end_time = time.time()
        time_dif = end_time - start_time
        return timedelta(seconds=int(round(time_dif)))

    def evaluate(self, model, data_iter, test=False):
        model.eval()
        loss_total = 0
        predict_all = np.array([], dtype=int)
        labels_all = np.array([], dtype=int)
        if test:
            print('test...')
        else:
            print('evaluate...')
        with torch.no_grad():
            for texts, labels in tqdm(data_iter, total=len(data_iter)):
                outputs = model(texts)
                loss = F.cross_entropy(outputs, labels)
                loss_total += loss
                labels = labels.data.cpu().numpy()
                predic = torch.max(outputs.data, 1)[1].cpu().numpy()
                labels_all = np.append(labels_all, labels)
                predict_all = np.append(predict_all, predic)

        acc = metrics.accuracy_score(labels_all, predict_all)
        if test:
            report = metrics.classification_report(
                labels_all, predict_all, digits=4)
            confusion = metrics.confusion_matrix(labels_all, predict_all)
            return acc, loss_total / len(data_iter), report, confusion
        return acc, loss_total / len(data_iter)

    def test(self, model, test_iter):
        # test
        model.load_state_dict(torch.load(self.save_path))
        model.eval()
        start_time = time.time()
        test_acc, test_loss, test_report, test_confusion = self.evaluate(model, test_iter, test=True)
        msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
        print(msg.format(test_loss, test_acc))
        print("Precision, Recall and F1-Score...")
        print(test_report)
        print("Confusion Matrix...")
        print(test_confusion)
        time_dif = self.get_time_dif(start_time)
        print("Time usage:", time_dif)

    def train(self, model, train_iter, dev_iter, test_iter, progress=gr.Progress()):
        start_time = time.time()
        model.train()
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
        # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=self.learning_rate,
                             warmup=0.05,
                             t_total=len(train_iter) * self.num_epochs)
        # total_batch = 0  # 记录进行到多少batch
        dev_best_loss = float('inf')
        last_improve = 0  # 记录上次验证集loss下降的batch数
        flag = False  # 记录是否很久没有效果提升
        model.train()
        gr.Info('开始训练')
        info_list = []

        for epoch in range(self.num_epochs):
            for i, (trains, labels) in enumerate(tqdm(train_iter)):
                outputs = model(trains)
                model.zero_grad()
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()

            if epoch % 5 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                train_acc, train_loss = self.evaluate(model, train_iter)
                dev_acc, dev_loss = self.evaluate(model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
                    torch.save(model.state_dict(), self.save_path)
                    improve = True
                    last_improve = epoch
                else:
                    improve = False
                time_dif = self.get_time_dif(start_time)
                info = {
                    'train_loss': train_loss.cpu().numpy().tolist(),
                    'train_acc': train_acc,
                    'val_loss': dev_loss.cpu().numpy().tolist(),
                    'val_acc': dev_acc,
                    'save': improve,
                    'cost': time_dif.total_seconds()}
                info_list.append(info)
                # msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                # print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                model.train()
            # total_batch += 1
            if epoch - last_improve > self.require_improvement:
                # 验证集loss超过15个epoch没下降，结束训练
                gr.Info("No optimization for a long time, auto-stopping...")
                flag = True
                break
            if flag:
                break
        self.test(model, test_iter)
        gr.Info('训练结束')
        return pd.DataFrame(info_list)

class Predict():
    def __init__(self, model_dir, save_dir):

        self.bert_path = model_dir
        self.model_name = 'bert'
        self.save_dir = save_dir
        self.cls2id, self.id2cls = self.get_cls2id()
        self.save_path = os.path.join(save_dir, self.model_name + '.ckpt')
        self.hidden_size = 768
        self.pad_size = 512
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.init_model()
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)


    def get_cls2id(self):
        cls2id_path = os.path.join(self.save_dir, 'cls2id.json')
        cls2id = read_json_file(cls2id_path)
        id2cls = {}
        for key, value in cls2id.items():
            id2cls[value] = key
        return cls2id, id2cls

    def init_model(self):
        model = Model(
            self.bert_path,
            self.hidden_size,
            len(list(self.cls2id.keys()))
        ).to(self.device)
        try:
            if torch.cuda.is_available():
                model.load_state_dict(torch.load(self.save_path))
            else:
                model.load_state_dict(torch.load(self.save_path, map_location=torch.device('cpu')))
        except Exception as e:
            print(f"加载模型时出错: {e}")
        model.eval()
        return model

    def predict_single(self, input_text):
        token = self.tokenizer.tokenize(input_text)
        token = [CLS] + token
        seq_len = len(token)
        mask = []
        token_ids = self.tokenizer.convert_tokens_to_ids(token)
        pad_size = self.pad_size
        contents = []
        if pad_size:
            if len(token) < pad_size:
                mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                token_ids += ([0] * (pad_size - len(token)))
            else:
                mask = [1] * pad_size
                token_ids = token_ids[:pad_size]
                seq_len = pad_size
        contents.append((token_ids, seq_len, mask))
        x = torch.LongTensor([token_ids]).to(self.device)
        seq_len = torch.LongTensor([seq_len]).to(self.device)
        mask = torch.LongTensor([mask]).to(self.device)
        outputs = self.model((x,seq_len, mask))
        scores = torch.sigmoid(outputs)
        scores = scores.detach().cpu().numpy().tolist()[0]
        score = scores[0]
        predic = torch.max(outputs.data, 1)[1].cpu()
        cls_id = int(predic[0].numpy())
        cls = self.id2cls[cls_id]

        return cls, cls_id, score

def train(csv_path, model_dir, batch_size, num_epochs, save_dir):
    text_class = TextClassBert(model_dir, batch_size, num_epochs, save_dir)
    train_data, val_data, test_data = text_class.get_train_val_test_data(csv_path)
    train_iter = text_class.build_iterator(train_data)
    val_iter = text_class.build_iterator(val_data)
    test_iter = text_class.build_iterator(test_data)
    model = Model(
        text_class.bert_path,
        text_class.hidden_size,
        len(list(text_class.cls2id.keys()))
    ).to(text_class.device)
    result_df = text_class.train(model, train_iter, val_iter, test_iter)
    return result_df

def predict(model_dir, save_dir, ask_str):
    if len(ask_str) == 0:
        gr.Warning('输入为空')
        return ''
    else:
        pre_class = Predict(model_dir, save_dir)
        # ask_str = '铂金卡在携程购买南航票证厦航实际承运航班，里程自动累计到了厦航，旅客首先联系南航，被告知需要自行找厦航删除里程累计记录，旅客联系厦航，厦航告知需要其提供一系列的证明材料。'
        # ask_str2 = '919机上安全演示包里没有安全须知卡 人工安全演示时无法给旅客展示安全须知 希望后续能补齐'
        cls, cls_id, score = pre_class.predict_single(ask_str)
        info = {
            '分类结果':cls,
            '类别id': cls_id,
            '置信度': score
        }
        return info

text_cls_train_route = Blueprint('text_cls_train', __name__)
@text_cls_train_route.route('/text_cls_train', methods=['POST'])
def text_recognition():
    # try:
    data = request.get_json()
    data_path = data.get('data_path')
    model_dir = data.get('model_dir')
    batch_size = data.get('batch_size')
    num_epochs = data.get('num_epochs')
    save_dir = data.get('save_dir')

    train_info = train(data_path, model_dir, batch_size, num_epochs, save_dir)
    train_info_json = train_info.to_json(orient='records', force_ascii=False)
    return jsonify({
        "train_info": train_info_json,
        'error': ''
    }), 200
    # except Exception as e:
    #     return jsonify({"error": str(e)}), 500

text_cls_predict_route = Blueprint('text_cls_predict', __name__)
@text_cls_predict_route.route('/text_cls_predict', methods=['POST'])
def text_recognition():
    try:
        data = request.get_json()
        model_dir = data.get('model_dir')
        save_dir = data.get('save_dir')
        pre_input_text = data.get('pre_input_text')

        pre_result = predict(model_dir, save_dir, pre_input_text)
        return jsonify({
            "pre_result": pre_result,
            'error': ''
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # text_class = TextClassBert()
    # train_data, val_data, test_data = text_class.get_train_val_test_data('./data/data.csv')
    # train_iter = text_class.build_iterator(train_data)
    # val_iter = text_class.build_iterator(val_data)
    # test_iter = text_class.build_iterator(test_data)
    # model = Model(
    #     text_class.bert_path,
    #     text_class.hidden_size,
    #     len(list(text_class.cls2id.keys()))
    # ).to(text_class.device)
    # text_class.train(model, train_iter, val_iter, test_iter)

    pass