from flask import request, jsonify, Blueprint

from local.function.text_clssification.build_data import train, predict

text_cls_train_route = Blueprint('text_cls_train', __name__)
@text_cls_train_route.route('/text_cls_train', methods=['POST'])
def text_recognition():
    # try:
    data = request.get_json()
    data_path = data.get('data_path')
    model_dir = data.get('model_dir')
    batch_size = data.get('batch_size')
    num_epochs = data.get('num_epochs')
    save_dir = data.get('save_dir')

    train_info = train(data_path, model_dir, batch_size, num_epochs, save_dir)
    train_info_json = train_info.to_json(orient='records', force_ascii=False)
    return jsonify({
        "train_info": train_info_json,
        'error': ''
    }), 200
    # except Exception as e:
    #     return jsonify({"error": str(e)}), 500

text_cls_predict_route = Blueprint('text_cls_predict', __name__)
@text_cls_predict_route.route('/text_cls_predict', methods=['POST'])
def text_recognition():
    try:
        data = request.get_json()
        model_dir = data.get('model_dir')
        save_dir = data.get('save_dir')
        pre_input_text = data.get('pre_input_text')

        pre_result = predict(model_dir, save_dir, pre_input_text)
        return jsonify({
            "pre_result": pre_result,
            'error': ''
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
