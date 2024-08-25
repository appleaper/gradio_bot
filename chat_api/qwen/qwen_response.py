from openai import OpenAI


def qwen_chat(textbox, show_history, system_state, history, model_type, model_name, steam_check_box):
    client = OpenAI(
        api_key="sk-96887b78af644ddf8ddcd831bfca13f0",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    if show_history is None:
        history = []
    if len(history) == 0:
        history = [{"role":"system","content":system_state}]
    elif len(history) >= 6:
        history = history[-6:]
        history[0] = {"role": "system", "content": system_state}
    else:
        history[0] = {"role":"system","content":system_state}
    history.append(
        {'role':'user', 'content':textbox}
    )
    completion = client.chat.completions.create(
        model=f"{model_type}-{model_name}",  # 更多模型请参见模型列表文档。
        messages=history,
    )
    response_message = completion.model_dump()['choices'][0]['message']['content']
    response_role = completion.model_dump()['choices'][0]['message']['role']
    response_dict = {'role': response_role, 'content':response_message}
    history.append(response_dict)
    show_history.append((textbox,response_message))
    return '', show_history, history