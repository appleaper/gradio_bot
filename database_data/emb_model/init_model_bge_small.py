import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

def model_init(
        model_path = '/home/pandas/snap/model/bge-small-zh-v1.5'
):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    model.eval()
    return model, tokenizer

def model_detect(sentences, model, tokenizer):
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
        sentence_embeddings = model_output[0][:, 0]
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings
