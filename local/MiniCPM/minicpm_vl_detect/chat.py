import os
import re
import copy
import torch
import hashlib
import functools
import traceback
import gradio as gr
from PIL import Image
import modelscope_studio as mgr
from decord import VideoReader, cpu
from transformers import AutoModel, AutoTokenizer

from local.MiniCPM.minicpm_vl_detect.utils_tool import encode_image, make_text, init_conversation, clear, create_multimodal_input, flushed, regenerate_button_clicked, check_mm_type, request
from local.MiniCPM.minicpm_vl_detect.few_shot import fewshot_add_demonstration, fewshot_request, select_chat_type
from utils.config_init import multimodal_model_path

model_name = 'MiniCPM-V 2.6'
introduction = """

## Features:
1. Chat with single image
2. Chat with multiple images
3. Chat with video
4. In-context few-shot learning

Click `How to use` tab to see examples.
"""

form_radio = {
    'choices': ['Beam Search', 'Sampling'],
    #'value': 'Beam Search',
    'value': 'Sampling',
    'interactive': True,
    'label': 'Decode Type'
}
ERROR_MSG = "发生错误，请重试"
MAX_NUM_FRAMES = 64

ModelPath = multimodal_model_path
def load_model(model_path):
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    return model, tokenizer

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

@cached_model_loader
def load_model_cached(model_path):
    return load_model(model_path)

def model_detect(image_path, question, model, tokenizer):
    image = Image.open(image_path).convert('RGB')
    msgs = [{'role': 'user', 'content': [image, question]}]
    res = model.chat(
        image=None,
        msgs=msgs,
        tokenizer=tokenizer
    )
    return res
def create_component(params, comp='Slider'):
    if comp == 'Slider':
        return gr.Slider(
            minimum=params['minimum'],
            maximum=params['maximum'],
            value=params['value'],
            step=params['step'],
            interactive=params['interactive'],
            label=params['label']
        )
    elif comp == 'Radio':
        return gr.Radio(
            choices=params['choices'],
            value=params['value'],
            interactive=params['interactive'],
            label=params['label']
        )
    elif comp == 'Button':
        return gr.Button(
            value=params['value'],
            interactive=True
        )

def encode_video(video):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    if hasattr(video, 'path'):
        vr = VideoReader(video.path, ctx=cpu(0))
    else:
        vr = VideoReader(video.file.path, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    if len(frame_idx)>MAX_NUM_FRAMES:
        frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
    video = vr.get_batch(frame_idx).asnumpy()
    video = [Image.fromarray(v.astype('uint8')) for v in video]
    video = [encode_image(v) for v in video]
    print('video frames:', len(video))
    return video

def encode_mm_file(mm_file):
    if check_mm_type(mm_file) == 'image':
        return [encode_image(mm_file)]
    if check_mm_type(mm_file) == 'video':
        return encode_video(mm_file)
    return None

def encode_message(_question):
    files = _question.files
    question = _question.text
    pattern = r"\[mm_media\]\d+\[/mm_media\]"
    matches = re.split(pattern, question)
    message = []
    if len(matches) != len(files) + 1:
        gr.Warning("Number of Images not match the placeholder in text, please refresh the page to restart!")
    assert len(matches) == len(files) + 1

    text = matches[0].strip()
    if text:
        message.append(make_text(text))
    for i in range(len(files)):
        message += encode_mm_file(files[i])
        text = matches[i + 1].strip()
        if text:
            message.append(make_text(text))
    return message

def count_video_frames(_context):
    num_frames = 0
    for message in _context:
        for item in message["content"]:
            #if item["type"] == "image": # For remote call
            if isinstance(item, Image.Image):
                num_frames += 1
    return num_frames

def chat(img, msgs, ctx, params=None, vision_hidden_states=None):
    model, tokenizer = load_model_cached(ModelPath)
    try:
        if msgs[-1]['role'] == 'assistant':
            msgs = msgs[:-1] # remove last which is added for streaming
        # print('msgs:', msgs)
        answer = model.chat(
            image=None,
            msgs=msgs,
            tokenizer=tokenizer,
            **params
        )
        if params['stream'] is False:
            res = re.sub(r'(<box>.*</box>)', '', answer)
            res = res.replace('<ref>', '')
            res = res.replace('</ref>', '')
            res = res.replace('<box>', '')
            answer = res.replace('</box>', '')
        # print('answer:')
        for char in answer:
            # print(char, flush=True, end='')
            yield char
    except Exception as e:
        print(e)
        traceback.print_exc()
        yield ERROR_MSG

def respond(_chat_bot, _app_cfg):
    if len(_app_cfg) == 0:
        yield (_chat_bot, _app_cfg)
    elif _app_cfg['images_cnt'] == 0 and _app_cfg['videos_cnt'] == 0:
        yield (_chat_bot, _app_cfg)
    else:
        _question = _chat_bot[-1][0]
        _context = _app_cfg['ctx'].copy()
        _context.append({'role': 'user', 'content': encode_message(_question)})

        videos_cnt = _app_cfg['videos_cnt']

        # if params_form == 'Beam Search':
        #     params = {
        #         'sampling': False,
        #         'stream': False,
        #         'num_beams': 3,
        #         'repetition_penalty': 1.2,
        #         "max_new_tokens": 2048
        #     }
        # else:
        params = {
            'sampling': True,
            'stream': True,
            'top_p': 0.8,
            'top_k': 100,
            'temperature': 0.7,
            'repetition_penalty': 1.05,
            "max_new_tokens": 2048
        }
        params["max_inp_length"] = 4352  # 4096+256

        if videos_cnt > 0:
            # params["max_inp_length"] = 4352 # 4096+256
            params["use_image_id"] = False
            params["max_slice_nums"] = 1 if count_video_frames(_context) > 16 else 2

        gen = chat("", _context, None, params)

        _context.append({"role": "assistant", "content": [""]})
        _chat_bot[-1][1] = ""

        for _char in gen:
            _chat_bot[-1][1] += _char
            _context[-1]["content"][0] += _char
            yield (_chat_bot, _app_cfg)

        _app_cfg['ctx'] = _context
        yield (_chat_bot, _app_cfg)

def clear_video_memory():
    torch.cuda.empty_cache()
def minicpm_ui():
    with gr.Tab(model_name):
        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                gr.Markdown(value=introduction)
                # params_form = create_component(form_radio, comp='Radio')
                regenerate = create_component({'value': '重新生成'}, comp='Button')
                clear_button = create_component({'value': '清除历史'}, comp='Button')
                clear_memory_button = create_component({'value': '清空显存缓存'}, comp='Button')
                clear_memory_button.click(clear_video_memory,None,None)

            with gr.Column(scale=3, min_width=500):
                app_session = gr.State({'sts': None, 'ctx': [], 'images_cnt': 0, 'videos_cnt': 0, 'chat_type': 'Chat'})
                chat_bot = mgr.Chatbot(label=f"Chat with {model_name}", value=copy.deepcopy(init_conversation),
                                       height=560, flushing=False, bubble_full_width=False)

                with gr.Tab("Chat") as chat_tab:
                    txt_message = create_multimodal_input()
                    chat_tab_label = gr.Textbox(value="Chat", interactive=False, visible=False)

                    txt_message.submit(
                        request,
                        [txt_message, chat_bot, app_session],
                        [txt_message, chat_bot, app_session]
                    ).then(
                        respond,
                        [chat_bot, app_session],
                        [chat_bot, app_session]
                    )

                with gr.Tab("Few Shot") as fewshot_tab:
                    fewshot_tab_label = gr.Textbox(value="Few Shot", interactive=False, visible=False)
                    with gr.Row():
                        with gr.Column(scale=1):
                            image_input = gr.Image(type="filepath", sources=["upload"])
                        with gr.Column(scale=3):
                            user_message = gr.Textbox(label="User")
                            assistant_message = gr.Textbox(label="Assistant")
                            with gr.Row():
                                add_demonstration_button = gr.Button("Add Example")
                                generate_button = gr.Button(value="Generate", variant="primary")
                    add_demonstration_button.click(
                        fewshot_add_demonstration,
                        [image_input, user_message, assistant_message, chat_bot, app_session],
                        [image_input, user_message, assistant_message, chat_bot, app_session]
                    )
                    generate_button.click(
                        fewshot_request,
                        [image_input, user_message, chat_bot, app_session],
                        [image_input, user_message, assistant_message, chat_bot, app_session]
                    ).then(
                        respond,
                        [chat_bot, app_session],
                        [chat_bot, app_session]
                    )

                chat_tab.select(
                    select_chat_type,
                    [chat_tab_label, app_session],
                    [app_session]
                )
                chat_tab.select( # do clear
                    clear,
                    [txt_message, chat_bot, app_session],
                    [txt_message, chat_bot, app_session, image_input, user_message, assistant_message]
                )
                fewshot_tab.select(
                    select_chat_type,
                    [fewshot_tab_label, app_session],
                    [app_session]
                )
                fewshot_tab.select( # do clear
                    clear,
                    [txt_message, chat_bot, app_session],
                    [txt_message, chat_bot, app_session, image_input, user_message, assistant_message]
                )
                chat_bot.flushed(
                    flushed,
                    outputs=[txt_message]
                )
                regenerate.click(
                    regenerate_button_clicked,
                    [chat_bot, app_session],
                    [txt_message, image_input, user_message, assistant_message, chat_bot, app_session]
                ).then(
                    respond,
                    [chat_bot, app_session],
                    [chat_bot, app_session]
                )
                clear_button.click(
                    clear,
                    [txt_message, chat_bot, app_session],
                    [txt_message, chat_bot, app_session, image_input, user_message, assistant_message]
                )


    with gr.Tab("How to use"):
        with gr.Column():
            with gr.Row():
                image_example = gr.Image(
                    value="/home/pandas/snap/code/RapidOcr/local/MiniCPM/image/m_bear2.gif",
                    label='1. Chat with single or multiple images', interactive=False, width=400,
                    elem_classes="example")
                example2 = gr.Image(
                    value="/home/pandas/snap/code/RapidOcr/local/MiniCPM/image/video2.gif",
                    label='2. Chat with video', interactive=False, width=400, elem_classes="example")
                example3 = gr.Image(
                    value="/home/pandas/snap/code/RapidOcr/local/MiniCPM/image/fshot.gif",
                    label='3. Few shot', interactive=False, width=400, elem_classes="example")