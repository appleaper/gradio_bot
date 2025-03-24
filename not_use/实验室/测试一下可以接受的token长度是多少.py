import time
from local.qwen.qwen_api import qwen_model_init, qwen_model_detect

model_path = r'C:\use\model\qwen2.5-7b-instruct'
model, tokenizer = qwen_model_init(model_path)
start = time.time()
input_txt = '我将提供一串数字，你需要告诉我最后一个数字是什么。'
for i in range(1500):
    input_txt += str(i) + ','
print(len(input_txt))
messages=[{'role': 'user', 'content': input_txt}]
res = qwen_model_detect(messages, model, tokenizer)
print(res)
print(f'cost:{time.time() - start}s')