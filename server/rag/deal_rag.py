from flask import request, jsonify, Blueprint
from local.rag.deal_rag import DealRag


rag_class = DealRag()

delete_article_route = Blueprint('delete_article', __name__)
@delete_article_route.route('/delete_article', methods=['POST'])
def text_recognition():
    try:
        data = request.get_json()
        required_params = ['rag_database_type', 'rag_embedding_type', 'rag_checkboxgroup', 'config_info']
        for param in required_params:
            if not data.get(param):
                return jsonify({"error": f"{param} 是必需的参数"}), 400
        rag_database_type = data.get('rag_database_type')
        rag_embedding_type = data.get('rag_embedding_type')
        rag_checkboxgroup = data.get('rag_checkboxgroup')
        config_info = data.get('config_info')

        config_info = rag_class.delete_article(rag_database_type, rag_embedding_type, rag_checkboxgroup, config_info)
        print(config_info)
        print(type(config_info))
        return jsonify({
            "config_info": config_info
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    pass