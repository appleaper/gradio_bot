import os
import pandas as pd
from flask import request, jsonify
from local.model.other_model.aigot import OCR_AiGot
from flask import Blueprint


ocr_class = OCR_AiGot()

analyze_images_route = Blueprint('analyze_images', __name__)
@analyze_images_route.route('/analyze_images', methods=['POST'])
def analyze_images():
    try:
        data = request.get_json()
        file_path_list = data.get('file_path_list')
        model_dir = data.get('model_dir')

        if not file_path_list or not model_dir:
            return jsonify({"error": "file_path_list 和 model_dir 是必需的参数"}), 400

        project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        tmp_dir = os.path.join(project_dir, 'data', 'tmp')
        os.makedirs(tmp_dir, exist_ok=True)
        save_path = os.path.join(tmp_dir, 'result.csv')

        ocr_class.init_mdoel(model_dir)
        out_list = []
        ocr_str = ''

        for index, file_path in enumerate(file_path_list):
            info = {}
            ocr_result = ocr_class.parse_image(file_path)
            filename = os.path.basename(file_path)
            file_name, suffix = os.path.splitext(filename)
            info['file_name'] = file_name
            info['result'] = ocr_result
            out_list.append(info)
            if index == 0:
                ocr_str += ocr_result

        df = pd.DataFrame(out_list)
        df.to_csv(save_path, encoding='utf-8', index=False)
        ocr_class.unload_model()

        return jsonify({
            "empty_list": [],
            "ocr_str": ocr_str,
            "save_path": save_path,
            "first_file_path": file_path_list[0]
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500