
from flask import request, jsonify, Blueprint
from local.function.paddle_ocr_torch.predict_cls import imgs_cls_predict

direction_judgment_route = Blueprint('direction_judgment', __name__)
@direction_judgment_route.route('/direction_judgment', methods=['POST'])
def direction_judgment():
    try:
        data = request.get_json()
        img_path = data.get('cls_img_path')
        model_path = data.get('cls_model_type')

        if not img_path or not model_path:
            return jsonify({"error": "cls_img_path 和 cls_model_type 是必需的参数"}), 400

        msg, [] = imgs_cls_predict(img_path, model_path)
        return jsonify({
            "cls_result": msg,
            'cls_img_path': [],
            'error': ''
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
