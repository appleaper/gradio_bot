from flask import Flask
from server.rag.parse.llm_ocr_server import analyze_images_route
from server.function.doclayout_yolo.analysis_of_plate_surface_server import psa_analysis_route
from server.function.paddle_ocr_torch.predict_cls_server import direction_judgment_route
from server.function.paddle_ocr_torch.predict_det_server import text_detection_route
from server.function.paddle_ocr_torch.predict_rec_server import text_recognition_route
from server.function.paddle_ocr_torch.predict_system_server import small_model_text_recognition_route
from server.function.text_clssification.build_data import text_cls_train_route, text_cls_predict_route
from server.rag.parse.voice_parse import stand_alone_speech_route
from server.rag.deal_rag import delete_article_route

app = Flask(__name__)
app.register_blueprint(analyze_images_route)
app.register_blueprint(psa_analysis_route)
app.register_blueprint(direction_judgment_route)
app.register_blueprint(text_detection_route)
app.register_blueprint(text_recognition_route)
app.register_blueprint(small_model_text_recognition_route)
app.register_blueprint(text_cls_train_route)
app.register_blueprint(text_cls_predict_route)
app.register_blueprint(stand_alone_speech_route)
app.register_blueprint(delete_article_route)

if __name__ == '__main__':
    app.run(port=4500)