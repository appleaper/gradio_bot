import json
import requests
import pandas as pd
import gradio as gr
from bs4 import BeautifulSoup
from utils.tool import load_data

def get_search_web_info(query):
    url = "https://api.bochaai.com/v1/web-search"

    payload = json.dumps({
      "query": query,
      "summary": True,
      "count": 10,
      "page": 1
    })

    headers = {
      'Authorization': 'Bearer sk-bc7b769ac896462a9ebb0c2759a1f3a7',
      'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    return response.json()

def get_webpage_info(url):
    try:
        # 发送GET请求获取网页内容
        response = requests.get(url)
        # 检查请求是否成功（状态码为200表示成功）
        if response.status_code == 200:
            # 设置网页内容的编码为 apparent_encoding
            response.encoding = response.apparent_encoding
            soup = BeautifulSoup(response.text, 'html.parser')
            # 获取网页标题
            title = soup.title.string if soup.title else "无标题"
            if title is None:
                title = ''
            # 获取网页文本内容（去除标签）
            text_content = soup.get_text().strip()
            return title, text_content
        else:
            print(f"请求失败，状态码: {response.status_code}")
            return '', ''
    except requests.RequestException as e:
        print(f"请求过程中出现错误: {e}")
        return '', ''

def deal_search_web_info(info):
    if info['code']==200:
        data = info['data']
        webPageValue_list = data['webPages']['value']
        info_list = []
        for info_i in webPageValue_list:
            url = info_i['displayUrl']
            title, content = get_webpage_info(url)
            info1 = title + content
            info1 = info1.replace('\n\n', '')
            info = {}
            if len(info1) >10:
                info['url'] = url
                info['info'] = info1
                info_list.append(info)
            else:
                continue
        info_df = pd.DataFrame(info_list)
        info_str = ''.join(info_df['info'].values)
        return info_str
    else:
        gr.Warning('无法联网')
        return pd.DataFrame([])

def online_search(query):
    search_info = get_search_web_info(query)
    # data = load_data(r'C:\use\code\RapidOcr_small\data\temp.pkl')
    # search_info = data['info']
    info_df = deal_search_web_info(search_info)

    return info_df


if __name__ == '__main__':
    query = '特朗普最新关税政策'
    online_search('123')