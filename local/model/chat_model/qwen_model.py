

import torch
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


class QwenChatModel():
    def __init__(self, device='cuda'):
        self.model_name = ''
        self.device_str = device
        self.platform = 'local'
        self.check_device(device)

    def check_device(self, device):
        if device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def init_model(self, model_dir):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype="auto",
            device_map="auto"
        )

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id[0] if isinstance(tokenizer.eos_token_id,
                                                                             list) else tokenizer.eos_token_id
        self.tokenizer = tokenizer

    def model_detect(self, messages):
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": messages}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # print(response)

        return response

    def unload_model(self):
        torch.cuda.empty_cache()  # 清空未使用的缓存
        torch.cuda.ipc_collect()  # 收集并释放未使用的 IPC 内存

    def model_stream_detect(self, messages):
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )  # 将对话格式转换为模型输入
        model_inputs = self.tokenizer([input_ids], return_tensors="pt").to(self.model.device)  # 编码输入并移至 GPU
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,  # 跳过提示文本
            skip_special_tokens=True  # 忽略特殊标记
        )
        # 使用线程来生成文本
        generation_kwargs = dict(model_inputs, streamer=streamer, max_new_tokens=8192)  # 设置生成参数
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)  # 在单独的线程中调用生成函数
        thread.start()

        return streamer


        # # 生成输入和注意力掩码
        # inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to(self.model.device)
        # attention_mask = inputs.ne(self.tokenizer.pad_token_id).long()
        #
        # # 创建一个流处理器
        # streamer = TextStreamer(self.tokenizer)
        #
        # # 生成文本并流式输出
        # res = self.model.generate(inputs, attention_mask=attention_mask, streamer=streamer, max_new_tokens=512)
        # for text in res:
        #     print(text)


if __name__ == '__main__':
    # model_name = 'Qwen/Qwen2.5-0.5B-Instruct'
    model_dir = r'C:\use\model\Qwen25_05B_Instruct'
    input_str = '天空为什么是蓝色的'
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": input_str}
    ]
    model_class = QwenChatModel('cuda')
    model_class.init_model(model_dir)
    # model_class.model_detect(input_str)
    # print('*'*50)
    streamer = model_class.model_stream_detect(input_str)
    for new_text in streamer:  # 从 streamer 中逐段获取生成的文本
        print(new_text)
    # model, tokenizer = model_class.init_model(model_dir)
    # history = [{'role':'user', 'content':f'{input_str}'}]
    # # model_class.model_stream_detect(input_str)
    # # print('*'*50)
    # model_class.model_stream_detect(history, model, tokenizer)

    # from transformers import AutoModelForCausalLM, AutoTokenizer

    # model_name = "Qwen/Qwen2.5-0.5B-Instruct"




