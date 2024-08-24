from PIL import Image, ImageDraw
from rapidocr_onnxruntime import RapidOCR

engine = RapidOCR()

img_path = '/home/pandas/snap/image/High_EQ_Sales_Course/16.jpg'
result, elapse = engine(img_path)
image = Image.open(img_path)
draw = ImageDraw.Draw(image)
out_str = ''
for i in result:
    points = []
    for j in i[0]:
        x,y = j[0], j[1]
        points.append((x,y))
    draw.line(points, fill='green', width=3)
    out_str += i[1] + '\n'
image.show()
print(out_str)