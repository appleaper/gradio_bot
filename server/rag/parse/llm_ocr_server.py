
from flask import request, jsonify
from flask import Blueprint
from local.rag.parse.pdf_parse import llm_ocr



analyze_images_route = Blueprint('analyze_images', __name__)
@analyze_images_route.route('/analyze_images', methods=['POST'])
def analyze_images():
    try:
        data = request.get_json()
        file_path_list = data.get('file_path_list')
        model_dir = data.get('model_dir')

        if not file_path_list or not model_dir:
            return jsonify({"error": "file_path_list 和 model_dir 是必需的参数"}), 400

        ocr_str, save_path, file_path_list = llm_ocr(model_dir, file_path_list)

        return jsonify({
            "empty_list": [],
            "ocr_str": ocr_str,
            "save_path": save_path,
            "first_file_path": file_path_list[0]
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500