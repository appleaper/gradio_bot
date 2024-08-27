from PIL import Image, ImageDraw, ImageFont
from rapidocr_onnxruntime import RapidOCR
from config import conf_yaml

font_path = conf_yaml['ocr']['font_path']
engine = RapidOCR()
def ocr_detect(img_path):
    result, elapse = engine(img_path)
    image = Image.open(img_path)
    return image, result

def show_result(img_path):
    image, ocr_result = ocr_detect(img_path)
    draw_image = Image.new("RGB", (image.width, image.height), "white")
    draw = ImageDraw.Draw(draw_image)
    font = ImageFont.truetype(font_path, 40)

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