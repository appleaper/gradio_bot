import requests
import json

API_KEY = "mxXMsLWlRtjys72p8O68ZnZy"
SECRET_KEY = "FafeXspZYxhjMCnUWWdnmddgexYhTRRe"

def get_access_token():
    """
    使用 AK，SK 生成鉴权签名（Access Token）
    :return: access_token，或是None(如果错误)
    """
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
    return str(requests.post(url, params=params).json().get("access_token"))

def baidu_api(history=[], role_set='', model_name='ernie_speed'):
    url = f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/{model_name}?access_token=" + get_access_token()
    payload = json.dumps({
        "messages": history,
        "system": role_set
    })
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    response_str = response.json()['result']
    response_role = 'assistant'
    return response_str, response_role

def baidu_chat(textbox, show_history, system_state, history, model_type, model_name, steam_check_box):
    if show_history is None:
        history = []
    if len(history) >= 6:
        history = history[-6:]
    history.append(
        {'role':'user', 'content':textbox}
    )
    response_message, response_role = baidu_api(history, system_state, model_name)
    response_dict = {'role': response_role, 'content':response_message}
    history.append(response_dict)
    show_history.append((textbox,response_message))
    return '', show_history, history

if __name__ == '__main__':
    history = [
        {
            "role": "user",
            "content": "你好"
        }
    ]
    role_set = "你是一个有用的助手"
    model_name = 'ernie-speed-128k'      # ernie_speed
    response_str = baidu_chat(history, role_set, model_name)
    print(response_str)