import torch
from transformers import AutoTokenizer, AutoModel


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def cls_pooling(model_output):
    return model_output[0][:, 0]

def init_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    model.eval()
    return model, tokenizer

def model_detect(texts1, model, tokenizer):
    with torch.no_grad():
        inputs1 = tokenizer(texts1, padding=True, truncation=True, return_tensors='pt')
        model_output1 = model(**inputs1)
        embs1 = cls_pooling(model_output1)
        embs1 = torch.nn.functional.normalize(embs1, p=2, dim=1).numpy()
    return embs1