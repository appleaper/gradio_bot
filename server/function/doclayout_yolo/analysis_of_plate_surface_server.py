from local.function.doclayout_yolo.analysis_of_plate_surface_server import psa_analysis
from flask import request, jsonify, Blueprint


psa_analysis_route = Blueprint('psa_analysis_server', __name__)
@psa_analysis_route.route('/psa_analysis_server', methods=['POST'])
def psa_analysis_server():
    try:
        data = request.get_json()
        img_path = data.get('psa_img_path')
        model_path = data.get('psa_model_path')

        if not img_path or not model_path:
            return jsonify({"error": "img_path 和 model_path 是必需的参数"}), 400

        save_path = psa_analysis(img_path, model_path)

        return jsonify({
            "save_path": save_path,
            'error': ''
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500