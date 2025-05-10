import os
import cv2
from doclayout_yolo import YOLOv10
from flask import request, jsonify, Blueprint


psa_analysis_route = Blueprint('psa_analysis', __name__)
@psa_analysis_route.route('/psa_analysis', methods=['POST'])
def psa_analysis():
    try:
        data = request.get_json()
        img_path = data.get('psa_img_path')
        model_path = data.get('psa_model_path')

        if not img_path or not model_path:
            return jsonify({"error": "img_path 和 model_path 是必需的参数"}), 400

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

        return jsonify({
            "save_path": save_path,
            'error': ''
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500