import os
import cv2
from doclayout_yolo import YOLOv10

def psa_analysis(img_path, model_path):
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    temp_dir = os.path.join(project_dir, 'data', 'tmp')
    os.makedirs(temp_dir, exist_ok=True)
    save_path = os.path.join(temp_dir, 'result.jpg')

    model = YOLOv10(model_path)
    det_res = model.predict(
        img_path,  # Image to predict
        imgsz=1024,  # Prediction image size
        conf=0.2,  # Confidence threshold
        device="cuda:0"  # Device to use (e.g., 'cuda:0' or 'cpu')
    )

    annotated_frame = det_res[0].plot(pil=True, line_width=5, font_size=20)
    cv2.imwrite(save_path, annotated_frame)
    return save_path