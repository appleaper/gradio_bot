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

# def model_detect(model_path='/home/pandas/snap/model/bge-small-zh-v1.5'):
#     tokenizer = AutoTokenizer.from_pretrained(model_path)
#     model = AutoModelForSequenceClassification.from_pretrained(model_path)
#     model.eval()
#
#     pairs = [
#         ['what is panda?', 'hi'],
#         ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']]
#     with torch.no_grad():
#         inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
#         scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
#         print(scores)

if __name__ == '__main__':
    model_detect()