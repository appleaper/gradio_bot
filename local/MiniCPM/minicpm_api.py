from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

if torch.cuda.is_available():
    device = "cuda"  # the device to load the model onto
else:
    device = 'cpu'
def minicpm_model_init(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=device,
                                                 trust_remote_code=True)
    return model, tokenizer

def minicpm_model_detect(messages, model, tokenizer):
    model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(device)

    model_outputs = model.generate(
        model_inputs,
        max_new_tokens=1024,
        top_p=0.7,
        temperature=0.7
    )

    output_token_ids = [
        model_outputs[i][len(model_inputs[i]):] for i in range(len(model_inputs))
    ]

    responses = tokenizer.batch_decode(output_token_ids, skip_special_tokens=True)[0]
    return responses