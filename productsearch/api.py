from ninja import NinjaAPI
from ninja import Schema
from .util import (reset_word_tokenize)

from .search_algo import supersearch, embeddsearch, supersearch_debug, embedding_first_text_filter_later

class SearchPrompts(Schema):
    sprompt: str

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

    return search_response

@api.post("/product_supersearch_debug")
def debug(request, searchprompts: SearchPrompts):
    user_search_prompt_tokenize = reset_word_tokenize(user_search_prompt=searchprompts.sprompt)
    search_response = supersearch_debug(user_query=user_search_prompt_tokenize)

    return search_response
