from flask import request, jsonify, Blueprint
from local.function.paddle_ocr_torch.predict_system import imgs_sys_predict

small_model_text_recognition_route = Blueprint('small_model_text_recognition', __name__)
@small_model_text_recognition_route.route('/small_model_text_recognition', methods=['POST'])
def text_detection():
    try:
        data = request.get_json()
        sys_img_path = data.get('sys_img_path')
        sys_model_type = data.get('sys_model_type')

        if not sys_img_path or not sys_model_type:
            return jsonify({"error": "sys_img_path 和 sys_model_type 是必需的参数"}), 400

        sys_df_result_df, sys_img_result, sys_img_path = imgs_sys_predict(sys_img_path, sys_model_type)
        sys_df_result = sys_df_result_df.to_json(orient='records', force_ascii=False)
        return jsonify({
            "sys_df_result": sys_df_result,
            'sys_img_result': sys_img_result,
            'error': ''
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    img_path = r'C:\use\code\PaddleOCR2Pytorch-main\doc\imgs\00018069.jpg'
    model_name = 'method_one'
    imgs_sys_predict(img_path, model_name)
    # main(utility.parse_args())