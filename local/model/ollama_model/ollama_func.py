from ollama import Client
from utils.tool import singleton
from local.model.emb_base_model import EmbBase

@singleton
class OllamaClient(EmbBase):
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.client = self.check_connection()
        self.model_name = ''
        self.platform = 'ollama'

    def check_connection(self):
        '''看看连接正不正常'''
        client = Client(
            host=f'http://{self.host}:{self.port}',
        )
        return client

    def generate_text(self, model_name, messages, stream=True):
        '''流式输出'''
        return self.client.chat(model=model_name, messages=messages, stream=stream)

    def list_models(self):
        '''列出所有模型'''
        res = self.client.list()
        if 'models' in res:
            model_name_list = []
            for model_info in res['models']:
                model_name_list.append(model_info['model'])
            return model_name_list
        else:
            return []

    def parse_single_sentence(self, sentence, model_name):
        res = self.client.embed(model=model_name, input=sentence)
        return res.embeddings

    def model_stream_detect(self, message, model_name):
        stream = self.client.chat(model=model_name, messages=message, stream=True)
        return stream

    def unload_model(self):
        pass

    def load_model(self):
        pass


if __name__ == '__main__':
    # 创建 Ollama 客户端实例
    host = '127.0.0.1'
    port = 11434
    client = OllamaClient(host, port)
    client.check_connection()
    models = client.list_models()
    print(models)
    input_text = '为什么天空是蓝色的?'
    model_name = 'qwen2.5:0.5b'
    messages = [{'role': 'user', 'content': input_text}]
    stream = client.model_stream_detect(messages, model_name)
    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)

    # stream = client.generate_text(
    #     model_name='qwen2.5:0.5b',
    #     messages=[{'role': 'user', 'content': input_text}],
    #     stream=True
    # )
    # for chunk in stream:
    #     print(chunk['message']['content'], end='', flush=True)
    #
    # emb_model_name = 'bge-m3:latest'
    # res1 = client.parse_single_sentence(emb_model_name, input_text)
    # print(len(res1))
    # sentences = ['你好', 'apple']
    # res2 = client.parse_single_sentence(emb_model_name, sentences)
    # print(len(res2))