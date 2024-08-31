from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from simpletransformers.ner import NERModel

def embeddings_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained("../searchapi/nlpModels/embedding_recent/")
    model = AutoModelForMaskedLM.from_pretrained("../searchapi/nlpModels/embedding_recent/")
    # tokenizer = AutoTokenizer.from_pretrained("../searchapi/nlpModels/embeddings_model/")
    # model = AutoModelForMaskedLM.from_pretrained("../searchapi/nlpModels/embeddings_model/")

    model.to(device)

    return model, tokenizer

def ner_model():
    model = NERModel(
        "camembert", "../searchapi/nlpModels/token_clf", use_cuda=False
    )

    return model