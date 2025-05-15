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
        return jsonify({
            "config_info": config_info
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

add_article_route = Blueprint('add_article', __name__)
@add_article_route.route('/add_article', methods=['POST'])
def text_recognition():
    try:
        data = request.get_json()
        #rag_database_type, rag_embedding_type, is_same_group, knowledge_name, rag_upload_file, config_info
        required_params = ['rag_database_type', 'rag_embedding_type', 'is_same_group', 'rag_upload_file', 'config_info']
        for param in required_params:
            if not data.get(param):
                return jsonify({"error": f"{param} 是必需的参数"}), 400
        rag_database_type = data.get('rag_database_type')
        rag_embedding_type = data.get('rag_embedding_type')
        is_same_group = data.get('is_same_group')
        knowledge_name = data.get('knowledge_name')
        rag_upload_file = data.get('rag_upload_file')
        config_info = data.get('config_info')
        config_info = rag_class.add_article(rag_database_type, rag_embedding_type, is_same_group, knowledge_name, rag_upload_file, config_info)
        return jsonify({
            "config_info": config_info
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

add_article_group_route = Blueprint('add_article_group', __name__)
@add_article_group_route.route('/add_article_group', methods=['POST'])
def add_article_group():
    try:
        data = request.get_json()
        # kb_database_type, kb_embedding_type, selectable_documents_checkbox_group, new_kb_name_textbox, config_info
        required_params = ['kb_database_type', 'kb_embedding_type', 'selectable_documents_checkbox_group', 'new_kb_name_textbox', 'config_info']
        for param in required_params:
            if not data.get(param):
                return jsonify({"error": f"{param} 是必需的参数"}), 400
        kb_database_type = data.get('kb_database_type')
        kb_embedding_type = data.get('kb_embedding_type')
        selectable_documents_checkbox_group = data.get('selectable_documents_checkbox_group')
        new_kb_name_textbox = data.get('new_kb_name_textbox')
        config_info = data.get('config_info')
        group_info_json, username = rag_class.add_article_group(kb_database_type, kb_embedding_type, selectable_documents_checkbox_group, new_kb_name_textbox, config_info)
        return jsonify({
            "group_info_json": group_info_json,
            'username': username
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

delete_article_group_route = Blueprint('delete_article_group', __name__)
@delete_article_group_route.route('/delete_article_group', methods=['POST'])
def delete_article_group():
    try:
        data = request.get_json()
        # kb_database_type, kb_embedding_type, selectable_knowledge_bases_checkbox_group, config_info
        required_params = ['kb_database_type', 'kb_embedding_type', 'selectable_knowledge_bases_checkbox_group', 'config_info']
        for param in required_params:
            if not data.get(param):
                return jsonify({"error": f"{param} 是必需的参数"}), 400
        kb_database_type = data.get('kb_database_type')
        kb_embedding_type = data.get('kb_embedding_type')
        selectable_knowledge_bases_checkbox_group = data.get('selectable_knowledge_bases_checkbox_group')
        config_info = data.get('config_info')
        group_info_json, username = rag_class.delete_article_group(kb_database_type, kb_embedding_type, selectable_knowledge_bases_checkbox_group, config_info)
        return jsonify({
            "group_info_json": group_info_json,
            'username': username
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

select_article_group_route = Blueprint('select_article_group', __name__)
@select_article_group_route.route('/select_article_group', methods=['POST'])
def select_article_group():
    try:
        data = request.get_json()
        # search_database_type, search_embedding_type, search_kb_range, search_tok_k, search_text, config_info
        required_params = ['search_database_type', 'search_embedding_type', 'search_kb_range', 'search_tok_k', 'search_text','config_info']
        for param in required_params:
            if not data.get(param):
                return jsonify({"error": f"{param} 是必需的参数"}), 400
        search_database_type = data.get('search_database_type')
        search_embedding_type = data.get('search_embedding_type')
        search_kb_range = data.get('search_kb_range')
        search_tok_k = data.get('search_tok_k')
        search_text = data.get('search_text')
        config_info = data.get('config_info')
        df, config_info = rag_class.select_article_group(
            search_database_type, search_embedding_type, search_kb_range, search_tok_k, search_text, config_info
        )
        df_json = df.to_json(orient='records', force_ascii=False)
        return jsonify({
            "df_json": df_json,
            'config_info': config_info
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    pass