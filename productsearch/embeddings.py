from .models import Product, NameEmbeddingsNew, DescEmbeddingsNew
from .loadNLPmodel import embeddings_model
import torch
from pythainlp import word_tokenize
from tqdm import tqdm

class Document:
    def __init__(self, page_content):
        self.page_content = page_content

def split_text(text, chunk_size, chunk_overlap):
    words = word_tokenize(text, engine='newmm') 
    chunks = []
    for i in range(0, len(words), chunk_size - chunk_overlap):
        chunk = words[i:i + chunk_size]
        chunks.append(' '.join(chunk))

    return chunks

def split_documents(documents):
    all_chunks = []
    for document in documents:
        chunks = split_text(document.page_content, chunk_size=30, chunk_overlap=3)
        for chunk in chunks:
            all_chunks.append(Document(page_content=chunk))

    return all_chunks

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def createEmbeddings(text=None, model=None, tokenizer=None):
    input_tokenize = tokenizer(text, 
                               padding="max_length", 
                               truncation=True, 
                               max_length=512, return_tensors="pt")
    with torch.no_grad():
        text_embedd = model.bert(**input_tokenize).last_hidden_state[:, 0, :].detach().cpu().numpy()
        # text_embedd = model(**input_tokenize)
        # text_embedd = mean_pooling(text_embedd, input_tokenize['attention_mask'])

    return text_embedd


def run():
    model, tokenizer = embeddings_model()
    products = Product.objects.all()

    for product in tqdm(products):
        productName = product.productName
        productDes = product.productDes

        productName_embedd = createEmbeddings(text=productName, 
                                              model=model, 
                                              tokenizer=tokenizer)
        name_embedd = NameEmbeddingsNew(
            product=product,
            embedding_name = productName_embedd[0],
        )

        name_embedd.save()
        print("Save Name Embeddings")

        documents = [Document(page_content=productDes)]
        descs = split_documents(documents)
        
        desc_embeddings = []
        for desc in descs:
            productDes_embedd = createEmbeddings(text=desc.page_content, 
                                                 model=model, 
                                                 tokenizer=tokenizer)
            desc_embeddings.append(productDes_embedd[0])

        # Store embeddings and link to products
        for text, desc_embedding in zip(descs, desc_embeddings):
            embedding_entry = DescEmbeddingsNew(
                product=product,
                embedding_desc=desc_embedding,
                document=text.page_content
            )
            embedding_entry.save()
            print("Save Desc Embeddings!")
    
    print("Finish Embeddings")