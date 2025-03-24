import hashlib
import functools
import os.path
import fitz
import gradio as gr
import pandas as pd
from PIL import Image
from tqdm import tqdm
from utils.tool import generate_unique_filename
from transformers import AutoModel, AutoTokenizer
from utils.config_init import name2path, device_str, tmp_dir_path

def cached_model_loader(func):
    cache = {}
    @functools.wraps(func)
    def wrapper(model_path):
        # 计算输入的哈希值
        input_hash = hashlib.md5(model_path.encode()).hexdigest()
        if input_hash not in cache:
            # 如果输入的哈希值不在缓存中，则加载模型并缓存结果
            cache[input_hash] = func(model_path)
        return cache[input_hash]
    return wrapper

def ocr_pdf(pdf_path, model, tokenizer):
    pdf_document = fitz.open(pdf_path)
    result_list = []
    for page_num in tqdm(range(len(pdf_document)), total=len(pdf_document)):
        page = pdf_document.load_page(page_num)

        # 将页面转换为图片
        pix = page.get_pixmap()

        # 将pixmap转换为Pillow图像对象
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        output_path = generate_unique_filename('jpg')
        img.save(output_path)
        ocr_result = model.chat(tokenizer, output_path, ocr_type='format')
        if os.path.exists(output_path):
            # 删除文件
            os.remove(output_path)
        info = {}
        info['page_num'] = str(page_num)
        info['content'] = ocr_result
        result_list.append(info)
    return result_list

def ocr_image(image_path, model, tokenizer):
    ocr_result = model.chat(tokenizer, image_path, ocr_type='format')
    info = {}
    info['page_num'] = ''
    info['content'] = ocr_result
    return [info]

def load_model(model_name):
    if model_name == 'StepfunOcr':
        model_path = name2path['StepfunOcr']
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True, low_cpu_mem_usage=True, device_map=device_str,
                                          use_safetensors=True, pad_token_id=tokenizer.eos_token_id)
        if device_str.startswith('cuda'):
            model = model.eval().cuda()
        else:
            model = model.eval().cpu()
        return model, tokenizer
    else:
        gr.Error(f'{model_name} not support!')
        assert False, 'model name not support!'

@cached_model_loader
def load_model_cached(model_name):
    return load_model(model_name)

def get_result_image(img_paths):
    model, tokenizer = load_model_cached('StepfunOcr')
    all_result = []
    for filename_path in img_paths:
        filename, suffix = os.path.splitext(os.path.basename(filename_path))
        if suffix in ['.jpeg', '.jpg', '.png']:
            ocr_result_list = ocr_image(filename_path, model, tokenizer)
        elif suffix== '.pdf':
            ocr_result_list = ocr_pdf(filename_path, model, tokenizer)
        else:
            gr.Warning(f'{filename_path} not support!')
            continue
        all_result.extend(ocr_result_list)
    df = pd.DataFrame(all_result)
    save_path = os.path.join(tmp_dir_path, generate_unique_filename('csv'))
    df.to_csv(save_path, encoding='utf', index=False)
    return all_result[0]['content'], save_path
