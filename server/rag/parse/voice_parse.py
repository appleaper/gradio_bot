from flask import request, jsonify, Blueprint
from local.rag.parse.voice_parse import standalone_voice_analysis

stand_alone_speech_route = Blueprint('stand_alone_speech', __name__)
@stand_alone_speech_route.route('/stand_alone_speech', methods=['POST'])
def text_recognition():
    try:
        data = request.get_json()
        speech_recognition_file = data.get('speech_recognition_file')
        model_dir = data.get('model_dir')

        if not speech_recognition_file or not model_dir:
            return jsonify({"error": "speech_recognition_file 和 model_dir 是必需的参数"}), 400

        msg, [] = standalone_voice_analysis(speech_recognition_file, model_dir)
        return jsonify({
            "speech_recognition_output_text": msg,
            'speech_recognition_file': [],
            'error': ''
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    pass