
from flask import request, jsonify, Blueprint
from local.function.paddle_ocr_torch.predict_det import imgs_det_predict


text_detection_route = Blueprint('text_detection', __name__)
@text_detection_route.route('/text_detection', methods=['POST'])
def text_recognition():
    try:
        data = request.get_json()
        det_img_path = data.get('det_img_path')
        det_model_type = data.get('det_model_type')

        if not det_img_path or not det_model_type:
            return jsonify({"error": "det_img_path 和 det_model_type 是必需的参数"}), 400

        msg, [] = imgs_det_predict(det_img_path, det_model_type)
        return jsonify({
            "det_result": msg,
            'error': ''
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


