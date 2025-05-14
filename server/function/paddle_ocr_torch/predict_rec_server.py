
from flask import request, jsonify, Blueprint
from local.function.paddle_ocr_torch.predict_rec import imgs_rec_predict

text_recognition_route = Blueprint('text_recognition', __name__)
@text_recognition_route.route('/text_recognition', methods=['POST'])
def text_detection():
    try:
        data = request.get_json()
        rec_img_path = data.get('rec_img_path')
        rec_model_type = data.get('rec_model_type')

        if not rec_img_path or not rec_model_type:
            return jsonify({"error": "rec_img_path 和 rec_model_type 是必需的参数"}), 400

        msg, [] = imgs_rec_predict(rec_img_path, rec_model_type)
        return jsonify({
            "rec_text_result": msg,
            'error': ''
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    img_path = r'C:\use\code\PaddleOCR2Pytorch-main\doc\imgs_words\ch\word_1.jpg'
    model_name = 'ch_ptocr_v4_rec_server_infer'
    imgs_rec_predict(img_path, model_name)
    # main(utility.parse_args())