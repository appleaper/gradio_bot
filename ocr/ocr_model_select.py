import hashlib
import functools
import gradio as gr
from config import conf_yaml
from ocr.ocr_show import ocr_detect
# from rapidocr_onnxruntime import RapidOCR
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModel, AutoTokenizer
from utils.config_init import font_path


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

def load_model(model_name):
    # if model_name == 'RapidOCR':
    #     model = RapidOCR()
    #     return model, ''
    if model_name == 'StepfunOcr':
        model_path = name2path['StepfunOcr']
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True, low_cpu_mem_usage=True, device_map='cuda',
                                          use_safetensors=True, pad_token_id=tokenizer.eos_token_id)
        model = model.eval().cuda()
        return model, tokenizer
    else:
        gr.Error(f'{model_name} not support!')
        assert False, 'model name not support!'

@cached_model_loader
def load_model_cached(model_name):
    return load_model(model_name)

def ocr_detect_way(image_path, model_name):
    model, tokenizer = load_model_cached(model_name)
    if model_name == 'RapidOCR':
        ocr_result = ocr_detect(image_path, model)
    elif model_name == 'StepfunOcr':
        ocr_result = model.chat(tokenizer, image_path, ocr_type='format')
    else:
        gr.Error(f'{model_name} not support!')
        assert False, f'{model_name} not support!'
    return ocr_result

def get_result_image(img_path, model_name):
    image = Image.open(img_path)
    draw_image = Image.new("RGB", (image.width, image.height), "white")
    draw = ImageDraw.Draw(draw_image)
    font = ImageFont.truetype(font_path, 40)
    ocr_result = ocr_detect_way(img_path, model_name)
    if model_name == 'RapidOCR':
        out_str = ''
        for i in ocr_result:
            points = []
            for j in i[0]:
                x, y = j[0], j[1]
                points.append((x, y))
            text_position = (points[0][0], points[0][1] - 10)
            draw.line(points, fill='green', width=3)
            draw.text(text_position, i[1], fill='black', font=font)
            out_str += i[1] + '\n'
        return image, draw_image, out_str
    elif model_name == 'StepfunOcr':
        return image, image, ocr_result
    else:
        gr.Error(f'{model_name} not support!')
        assert False, f'{model_name} not support!'
