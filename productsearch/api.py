from ninja import NinjaAPI
from ninja import Schema
from .util import (reset_word_tokenize, 
                   get_all_product, 
                   response_formatting_frontend, 
                   get_product_to_compare_from_reactjs,
                   get_single_product_to_compare_from_reactjs)

from .search_algo import supersearch, embeddsearch, supersearch_debug, embedding_first_text_filter_later
from .chatGPT_request import generate_suggestion_normal, generate_suggestion_single_product, generate_any_suggestion

from random import randint

class SearchPrompts(Schema):
    sprompt: str

class ProductToCompare(Schema):
    productids: list

class ProductToSuggestion(Schema):
    productid: str

api = NinjaAPI()

@api.get("/hello")
def hello(request):
    return "I'm the Python text that will make you HELLO WORLD!"

@api.post("/product_supersearch")
def user_supersearch(request, searchprompts: SearchPrompts):
    user_search_prompt_tokenize = reset_word_tokenize(user_search_prompt=searchprompts.sprompt)
    search_response = supersearch(user_query=user_search_prompt_tokenize)

    return search_response

@api.post("/product_embeddsearch")
def user_embeddsearch(request, searchprompts: SearchPrompts):
    user_search_prompt_tokenize = reset_word_tokenize(user_search_prompt=searchprompts.sprompt)
    search_response = embeddsearch(user_query=user_search_prompt_tokenize)

    return search_response

@api.post("/product_embedd_first_text_later")
def user_embedding_first_text_filter_later(request, searchprompts: SearchPrompts):
    user_search_prompt_tokenize = reset_word_tokenize(user_search_prompt=searchprompts.sprompt)
    search_response = embedding_first_text_filter_later(user_query=user_search_prompt_tokenize)
    product_response = response_formatting_frontend(search_response)

    return product_response

@api.post("/product_supersearch_debug")
def debug(request, searchprompts: SearchPrompts):
    print(searchprompts.sprompt)
    user_search_prompt_tokenize = reset_word_tokenize(user_search_prompt=searchprompts.sprompt)
    search_response = supersearch_debug(user_query=user_search_prompt_tokenize)

    return search_response

@api.get("/getproduct")
def get_product(request):
    start_id = randint(0, 1800)
    end_id = randint(start_id, start_id+50)
    products = get_all_product()[start_id:end_id]
    products_response = response_formatting_frontend(products)

    return products_response

@api.post("/user_product_comparison")
def compare_product(request, producttocompare: ProductToCompare):
    productids_list = producttocompare.productids
    products_to_compare = get_product_to_compare_from_reactjs(productids_list=productids_list)
    products_response = response_formatting_frontend(products_to_compare)


    return products_response

@api.post("/product_suggestion")
def llm_suggestion(request, producttocompare: ProductToCompare):
    productids_list = producttocompare.productids
    products_to_compare = get_product_to_compare_from_reactjs(productids_list=productids_list)
    suggestion = generate_suggestion_normal(products_to_compare)
    suggestion_html = "<p>" + suggestion.replace(' ', '&nbsp;').replace('\n', '<br />').replace("**", "") + "</p>"

    return suggestion_html

@api.post("/product_single_suggestion")
def llm_single_suggestion(request, producttosuggestion: ProductToSuggestion):
    productid_list = [producttosuggestion.productid]
    products_to_suggestion = get_single_product_to_compare_from_reactjs(productid=productid_list)
    suggestion = generate_suggestion_single_product(products_to_suggestion)
    suggestion_html = "<p>" + suggestion.replace(' ', '&nbsp;').replace('\n', '<br />').replace("**", "") + "</p>"

    return suggestion_html

@api.post("/product_any_suggestion")
def llm_suggestion(request, searchprompts: SearchPrompts):
    user_prompt = searchprompts.sprompt
    suggestion = generate_any_suggestion(user_prompt)
    suggestion_html = "<p>" + suggestion.replace(' ', '&nbsp;').replace('\n', '<br />').replace("**", "") + "</p>"

    return suggestion_html
