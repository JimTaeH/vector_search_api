import scipy.stats
from .loadNLPmodel import embeddings_model, ner_model
from .util import (from_prediction, 
                   extract_integer_from_list, 
                   get_product_uuids, reset_word_tokenize)
from .word_dict import MyDictionary
import re
from django.db.models import Avg
from .models import Product, NameEmbeddings, DescEmbeddings
from pgvector.django import CosineDistance
from .embeddings import createEmbeddings
from scipy.spatial import distance
import numpy as np
import pandas as pd

model_NER = ner_model()
model_embedding, tokenizer_embedding = embeddings_model()
my_dict = MyDictionary()

def filter_product_by_price(price, my_dict, words_about_price, words_about_brand):
    # Calculate the average price
    average_price = Product.objects.aggregate(Avg('price'))['price__avg']
    #productID filterd by price
    product_uuids = get_product_uuids(price, my_dict, words_about_price, words_about_brand, average_price)
    product_by_price = Product.objects.filter(vector_product_id__in=product_uuids)

    return product_by_price

def filter_product_by_name(product_name_list=None, product_by_price=None):
    # product_name = "".join(product_name_list)
    product_name = product_name_list[0]
    product_by_name = product_by_price.filter(productName__contains=product_name)

    return product_by_name

def response_formatting(response_query_set=None):
    response_dict = {
        "name": [],
        "price": [],
        "link": [],
        "brand": []
    }

    for response_query in response_query_set:
        # response_product_id = response_query.product_id
        # reponse_product = Product.objects.filter(vector_product_id=response_product_id)
        product_name = response_query.productName
        product_price = response_query.price
        product_link = response_query.link
        product_brand = response_query.brand

        response_dict["name"].append(product_name)
        response_dict["price"].append(product_price)
        response_dict["link"].append(product_link)
        response_dict["brand"].append(product_brand)

    return response_dict

def description_embedding_search(product_query_set=None, user_search_prompt_embedd=None):
    query_results = {
        "product_id": [],
        "average_distance": [],
        "median_distance": []
    }

    for product in product_query_set:
        batch_distance = []
        product_desc_embedd = DescEmbeddings.objects.filter(product_id=product)
        
        for embedd in product_desc_embedd:
            embedd_distance = distance.cosine(user_search_prompt_embedd[0], embedd.embedding_desc)
            batch_distance.append(embedd_distance)
        
        query_results["product_id"].append(product.vector_product_id)
        query_results["average_distance"].append(np.mean(batch_distance))
        query_results["median_distance"].append(np.median(batch_distance))
    
    desc_embeddings_df = pd.DataFrame.from_dict(query_results)
    desc_embeddings_df = desc_embeddings_df.sort_values(by=["average_distance"], ascending=True)


    return desc_embeddings_df

def embedding_first_text_filter_later(user_query=None):
    NER_prompt_prediction, raw_ouputs = model_NER.predict([user_query])
    
    words_about_price, _ = from_prediction(NER_prompt_prediction,'3')
    str_about_price = "".join(words_about_price)
    price = extract_integer_from_list(words_about_price) 

    words_about_brand, _ = from_prediction(NER_prompt_prediction,'2') 

    words_about_product, list_product = from_prediction(NER_prompt_prediction,'0')
    str_about_product = " ".join(words_about_product)
    
    user_query = user_query.replace(" ", "")
    user_query_cut_price = user_query.replace(str_about_price, "").strip()
    user_query_cut_price = reset_word_tokenize(user_search_prompt=user_query_cut_price)
    print(user_query_cut_price)
    
    user_search_prompt_embedd = createEmbeddings(text=user_query_cut_price, 
                                                 model=model_embedding, 
                                                 tokenizer=tokenizer_embedding)
    
    product_by_price = filter_product_by_price(price=price, 
                                               my_dict=my_dict, 
                                               words_about_price=words_about_price, 
                                               words_about_brand=words_about_brand)
    
    product_name_embedd_filter = NameEmbeddings.objects.filter(product_id__in=product_by_price)

    product_name_embedd_similarity = product_name_embedd_filter.order_by(CosineDistance('embedding_name', 
                                                                                        user_search_prompt_embedd[0]))[0:30]
    filter_by_name_embedd_product_id = [x.product_id for x in product_name_embedd_similarity]
    product_filter_by_name_embedd = Product.objects.filter(vector_product_id__in=filter_by_name_embedd_product_id)
    print(len(product_filter_by_name_embedd))
    
    desc_embeddings_similarity_df = description_embedding_search(product_query_set=product_filter_by_name_embedd, 
                                                                 user_search_prompt_embedd=user_search_prompt_embedd)
    
    print(desc_embeddings_similarity_df.shape)
    product_desc_embedd_similarity = DescEmbeddings.objects.filter(product_id__in=desc_embeddings_similarity_df["product_id"].values.tolist())

    filter_by_desc_embedd_product_id = [x.product_id for x in product_desc_embedd_similarity]
    filter_by_desc_embedd_product_id = set(filter_by_desc_embedd_product_id)

    product_filter_by_embedd = Product.objects.filter(vector_product_id__in=filter_by_desc_embedd_product_id)

    print(list_product)
    product_by_name = filter_product_by_name(product_name_list=list_product, 
                                             product_by_price=product_filter_by_embedd)
    
    search_response = response_formatting(response_query_set=product_filter_by_embedd)

    response = {
        "results": search_response
    }

    return response

def supersearch(user_query=None):
    user_search_prompt_embedd = createEmbeddings(text=user_query, 
                                                 model=model_embedding, 
                                                 tokenizer=tokenizer_embedding)
    
    NER_prompt_prediction, raw_ouputs = model_NER.predict([user_query])

    words_about_price, _ = from_prediction(NER_prompt_prediction,'3') 
    price = extract_integer_from_list(words_about_price) 

    words_about_brand, _ = from_prediction(NER_prompt_prediction,'2') 

    words_about_product, list_product = from_prediction(NER_prompt_prediction,'0')
    str_about_product = " ".join(words_about_product)

    product_by_price = filter_product_by_price(price=price, 
                                               my_dict=my_dict, 
                                               words_about_price=words_about_price, 
                                               words_about_brand=words_about_brand)
    
    product_by_name = filter_product_by_name(product_name_list=list_product, 
                                             product_by_price=product_by_price)
    
    product_name_embedd_filter = NameEmbeddings.objects.filter(product_id__in=product_by_price)
    product_desc_embedd_filter = DescEmbeddings.objects.filter(product_id__in=product_by_price)
    
    product_name_embedd_similarity = product_name_embedd_filter.order_by(CosineDistance('embedding_name', 
                                                                                        user_search_prompt_embedd[0]))[:10]
    
    product_desc_embedd_similarity = product_desc_embedd_filter.order_by(CosineDistance('embedding_desc', 
                                                                                        user_search_prompt_embedd[0]))[:10]

    search_response_by_name = response_formatting(response_query_set=product_name_embedd_similarity)
    search_response_by_desc = response_formatting(response_query_set=product_desc_embedd_similarity)
    
    both_response = {
        "byNameEmbedd": search_response_by_name,
        "byDescEmbedd": search_response_by_desc
    }

    return both_response

def embeddsearch(user_query=None):
    user_search_prompt_embedd = createEmbeddings(text=user_query, 
                                                 model=model_embedding, 
                                                 tokenizer=tokenizer_embedding)
    product_name_embedd_similarity = NameEmbeddings.objects.order_by(CosineDistance('embedding_name', 
                                                                                    user_search_prompt_embedd[0]))
    
    product_desc_embedd_similarity = DescEmbeddings.objects.order_by(CosineDistance('embedding_desc', 
                                                                                    user_search_prompt_embedd[0]))
    
    print(len(product_name_embedd_similarity))
    print(len(product_desc_embedd_similarity))

    search_response_by_name = response_formatting(response_query_set=product_name_embedd_similarity[0:10])
    search_response_by_desc = response_formatting(response_query_set=product_desc_embedd_similarity[0:10])
    
    both_response = {
        "byNameEmbedd": search_response_by_name,
        "byDescEmbedd": search_response_by_desc
    }

    return both_response

def supersearch_debug(user_query=None):
    user_search_prompt_embedd = createEmbeddings(text=user_query, 
                                                 model=model_embedding, 
                                                 tokenizer=tokenizer_embedding)
    
    NER_prompt_prediction, raw_ouputs = model_NER.predict([user_query])

    words_about_price, _ = from_prediction(NER_prompt_prediction,'3') 
    price = extract_integer_from_list(words_about_price) 

    words_about_brand, _ = from_prediction(NER_prompt_prediction,'2') 

    words_about_product, list_product = from_prediction(NER_prompt_prediction,'0')
    str_about_product = " ".join(words_about_product)

    product_by_price = filter_product_by_price(price=price, 
                                               my_dict=my_dict, 
                                               words_about_price=words_about_price, 
                                               words_about_brand=words_about_brand)
    
    product_by_name = filter_product_by_name(product_name_list=list_product, 
                                             product_by_price=product_by_price)
    
    product_name_embedd_filter = NameEmbeddings.objects.filter(product_id__in=product_by_price)
    product_desc_embedd_filter = DescEmbeddings.objects.filter(product_id__in=product_by_price)
    
    product_name_embedd_similarity = product_name_embedd_filter.order_by(CosineDistance('embedding_name', 
                                                                                        user_search_prompt_embedd[0]))[:10]
    
    desc_embeddings_similarity_df = description_embedding_search(product_query_set=product_by_price, 
                                                                 user_search_prompt_embedd=user_search_prompt_embedd)
    print(desc_embeddings_similarity_df["product_id"].values.tolist()[0:10])
    product_desc_embedd_similarity = DescEmbeddings.objects.filter(product_id__in=desc_embeddings_similarity_df["product_id"].values.tolist()[0:10])

    search_response_by_name = response_formatting(response_query_set=product_name_embedd_similarity)
    search_response_by_desc = response_formatting(response_query_set=product_desc_embedd_similarity)
    
    both_response = {
        "byNameEmbedd": search_response_by_name,
        "byDescEmbedd": search_response_by_desc
    }

    return both_response
