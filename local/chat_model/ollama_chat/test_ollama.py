import json
import requests

base_url = "http://localhost:11434/api/generate"
data = {
    "model": "qwen2.5:0.5b",
    "prompt": "你是谁？",
    "stream": True
}

try:
    response = requests.post(base_url, json=data, stream=True)
    response.raise_for_status()
    for line in response.iter_lines():
        if line:
            result = line.decode('utf-8').strip()
            chunk = json.loads(result)
            if "response" in chunk:
                print(chunk["response"], end="", flush=True)
except requests.exceptions.RequestException as e:
    print(f"请求出错: {e}")