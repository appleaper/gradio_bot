import gradio as gr
from PIL import Image
from local.MiniCPM.minicpm_vl_detect.utils_tool import encode_image, make_text

def fewshot_add_demonstration(_image, _user_message, _assistant_message, _chat_bot, _app_cfg):
    ctx = _app_cfg["ctx"]
    message_item = []
    if _image is not None:
        image = Image.open(_image).convert("RGB")
        ctx.append({"role": "user", "content": [encode_image(image), make_text(_user_message)]})
        message_item.append({"text": "[mm_media]1[/mm_media]" + _user_message, "files": [_image]})
        _app_cfg["images_cnt"] += 1
    else:
        if _user_message:
            ctx.append({"role": "user", "content": [make_text(_user_message)]})
            message_item.append({"text": _user_message, "files": []})
        else:
            message_item.append(None)
    if _assistant_message:
        ctx.append({"role": "assistant", "content": [make_text(_assistant_message)]})
        message_item.append({"text": _assistant_message, "files": []})
    else:
        message_item.append(None)

    _chat_bot.append(message_item)
    return None, "", "", _chat_bot, _app_cfg

def fewshot_request(_image, _user_message, _chat_bot, _app_cfg):
    if _app_cfg["images_cnt"] == 0 and not _image:
        gr.Warning("Please chat with at least one image.")
        return None, '', '', _chat_bot, _app_cfg
    if _image:
        _chat_bot.append([
            {"text": "[mm_media]1[/mm_media]" + _user_message, "files": [_image]},
            ""
        ])
        _app_cfg["images_cnt"] += 1
    else:
        _chat_bot.append([
            {"text": _user_message, "files": [_image]},
            ""
        ])

    return None, '', '', _chat_bot, _app_cfg

def select_chat_type(_tab, _app_cfg):
    _app_cfg["chat_type"] = _tab
    return _app_cfg