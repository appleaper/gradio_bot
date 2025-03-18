import os
import copy
import gradio as gr
from PIL import Image
import modelscope_studio as mgr

init_conversation = [
    [
        None,
        {
            # The first message of bot closes the typewriter.
            "text": "You can talk to me now",
            "flushing": False
        }
    ],
]
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.mov', '.avi', '.flv', '.wmv', '.webm', '.m4v'}

def make_text(text):
    return text

def encode_image(image):
    if not isinstance(image, Image.Image):
        if hasattr(image, 'path'):
            image = Image.open(image.path).convert("RGB")
        else:
            image = Image.open(image.file.path).convert("RGB")
    # resize to max_size
    max_size = 448*16
    if max(image.size) > max_size:
        w,h = image.size
        if w > h:
            new_w = max_size
            new_h = int(h * max_size / w)
        else:
            new_h = max_size
            new_w = int(w * max_size / h)
        image = image.resize((new_w, new_h), resample=Image.BICUBIC)
    return image
    ## save by BytesIO and convert to base64
    #buffered = io.BytesIO()
    #image.save(buffered, format="png")
    #im_b64 = base64.b64encode(buffered.getvalue()).decode()
    #return {"type": "image", "pairs": im_b64}

def flushed():
    return gr.update(interactive=True)

def create_multimodal_input(upload_image_disabled=False, upload_video_disabled=False):
    '''
    创建了一个多模态输入界面，用户可以通过这个界面上传图片和视频。函数提供了灵活的配置选项，允许调用者控制上传按钮的启用/禁用状态。
    :param upload_image_disabled:   默认值为 False，表示上传图片按钮默认是启用的。
    :param upload_video_disabled:   默认值为 False，表示上传视频按钮默认是启用的。
    :return:
    '''
    return mgr.MultimodalInput(
        value=None,     # 用于设置初始值
        upload_image_button_props={
            'label': 'Upload Image',        # 按钮上显示的文本
            'disabled': upload_image_disabled,  # 按钮是否被禁用
            'file_count': 'multiple'        # 允许上传的文件数量，这里设置为 'multiple'，表示可以上传多个图片。
        },      # 设置上传图片按钮的属性
        upload_video_button_props={
            'label': 'Upload Video',        # 按钮上显示的文本
            'disabled': upload_video_disabled,  # 按钮是否被禁用
            'file_count': 'single'          # 允许上传的文件数量，这里设置为 'single'，表示只能上传一个视频。
        },      # 设置上传视频按钮的属性
        submit_button_props={
            'label': 'Submit'
        }   # 设置提交按钮的属性
    )

def clear(txt_message, chat_bot, app_session):
    txt_message.files.clear()
    txt_message.text = ''
    chat_bot = copy.deepcopy(init_conversation)
    app_session['sts'] = None
    app_session['ctx'] = []
    app_session['images_cnt'] = 0
    app_session['videos_cnt'] = 0
    return create_multimodal_input(), chat_bot, app_session, None, '', ''

def get_file_extension(filename):
    return os.path.splitext(filename)[1].lower()

def is_image(filename):
    return get_file_extension(filename) in IMAGE_EXTENSIONS

def is_video(filename):
    return get_file_extension(filename) in VIDEO_EXTENSIONS

def check_mm_type(mm_file):
    if hasattr(mm_file, 'path'):
        path = mm_file.path
    else:
        path = mm_file.file.path
    if is_image(path):
        return "image"
    if is_video(path):
        return "video"
    return None

def check_has_videos(_question):
    images_cnt = 0
    videos_cnt = 0
    for file in _question.files:
        if check_mm_type(file) == "image":
            images_cnt += 1
        else:
            videos_cnt += 1
    return images_cnt, videos_cnt

def request(_question, _chat_bot, _app_cfg):
    images_cnt = _app_cfg['images_cnt']
    videos_cnt = _app_cfg['videos_cnt']
    files_cnts = check_has_videos(_question)
    if files_cnts[1] + videos_cnt > 1 or (files_cnts[1] + videos_cnt == 1 and files_cnts[0] + images_cnt > 0):
        gr.Warning("Only supports single video file input right now!")
        return _question, _chat_bot, _app_cfg
    if files_cnts[1] + videos_cnt + files_cnts[0] + images_cnt <= 0:
        gr.Warning("Please chat with at least one image or video.")
        return _question, _chat_bot, _app_cfg
    _chat_bot.append((_question, None))
    images_cnt += files_cnts[0]
    videos_cnt += files_cnts[1]
    _app_cfg['images_cnt'] = images_cnt
    _app_cfg['videos_cnt'] = videos_cnt
    upload_image_disabled = videos_cnt > 0
    upload_video_disabled = videos_cnt > 0 or images_cnt > 0
    return create_multimodal_input(upload_image_disabled, upload_video_disabled), _chat_bot, _app_cfg

def regenerate_button_clicked(_chat_bot, _app_cfg):
    if len(_chat_bot) <= 1 or not _chat_bot[-1][1]:
        gr.Warning('No question for regeneration.')
        return None, None, '', '', _chat_bot, _app_cfg
    if _app_cfg["chat_type"] == "Chat":
        images_cnt = _app_cfg['images_cnt']
        videos_cnt = _app_cfg['videos_cnt']
        _question = _chat_bot[-1][0]
        _chat_bot = _chat_bot[:-1]
        _app_cfg['ctx'] = _app_cfg['ctx'][:-2]
        files_cnts = check_has_videos(_question)
        images_cnt -= files_cnts[0]
        videos_cnt -= files_cnts[1]
        _app_cfg['images_cnt'] = images_cnt
        _app_cfg['videos_cnt'] = videos_cnt

        _question, _chat_bot, _app_cfg = request(_question, _chat_bot, _app_cfg)
        return _question, None, '', '', _chat_bot, _app_cfg
    else:
        last_message = _chat_bot[-1][0]
        last_image = None
        last_user_message = ''
        if last_message.text:
            last_user_message = last_message.text
        if last_message.files:
            last_image = last_message.files[0].file.path
        _chat_bot[-1][1] = ""
        _app_cfg['ctx'] = _app_cfg['ctx'][:-2]
        return _question, None, '', '', _chat_bot, _app_cfg