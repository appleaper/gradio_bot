import time

import torch.cuda
from transformers import AutoTokenizer, AutoModelForCausalLM

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def llama3_model_init(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype="auto", device_map="auto"
    )
    return model, tokenizer
def llama3_model_detect(messages, tokenizer, model):
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        input_ids,
        max_new_tokens=8192,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)