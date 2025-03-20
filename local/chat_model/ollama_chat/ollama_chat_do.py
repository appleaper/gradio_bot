
from ollama import chat

def ollama_chat_do(history, parm_b):
    stream = chat(
        model=parm_b,
        messages=history,
        stream=True,
    )

    for chunk in stream:
        yield chunk['message']['content']