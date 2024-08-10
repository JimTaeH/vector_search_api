import re
from .models import Product
from pythainlp import word_tokenize

def from_prediction(data,val):
    concate_string_result = []
    list_of_key = []

    for sublist in data:
        concatenated_string = ""
        for dictionary in sublist:
            for key, value in dictionary.items():
                if value == val:  # Checking if the value is 3
                    concatenated_string += key
                    list_of_key.append(key)
        concate_string_result.append(concatenated_string.replace(" ",""))

    return concate_string_result, list_of_key

def extract_integer_from_list(lst):
    items = []
    for item in lst:
        if isinstance(item, int):  # Check if the item is already an integer
            items.append(item)
        elif isinstance(item, str):
            # Find all numbers in the string and convert them to integers
            found_numbers = re.findall(r'\d+', item)
            for number in found_numbers:
                items.append(int(number))
    return items  # Return the list of extracted integers

def get_product_uuids(price_in_list, my_dict_instance, about_price, about_brand, average_price):
    """Main function to get product_uuids based on different conditions."""

    def update_product_uuids(price_condition, price_value=None):
        """Helper function to update product_uuids based on given conditions."""
        query_kwargs = {'brand': about_brand[0]} if about_brand and about_brand[0] else {}
        if price_condition == 'cheaper':
            query_kwargs['price__lt'] = int(price_value)
        elif price_condition == 'expensive':
            query_kwargs['price__gt'] = int(price_value)
        return Product.objects.filter(**query_kwargs).values_list('vector_product_id', flat=True)

    def determine_price_bounds():
        """Determine the appropriate query based on the number of prices in list and price conditions."""
        if len(price_in_list) == 1:
            price_cond = 'cheaper' if my_dict_instance.is_cheaper(about_price) else 'expensive'
            return update_product_uuids(price_cond, price_in_list[0])
        elif len(price_in_list) == 2:
            product_uuids = Product.objects.filter(
                price__lt=int(max(price_in_list)),
                price__gt=int(min(price_in_list)),
                **({'brand': about_brand[0]} if about_brand and about_brand[0] else {})
            ).values_list('vector_product_id', flat=True)
            return product_uuids
        else:
            return update_product_uuids(None)  # No specific price condition is applied here

    # Main logic of get_product_uuids
    if len(price_in_list) > 0:
        return determine_price_bounds()
    else:

        if my_dict_instance.is_cheaper(about_price):
            average_condition = 'cheaper'
            average_price_adjusted = average_price * 0.8
        elif my_dict_instance.is_expensive(about_price):
            average_condition = 'expensive'
            average_price_adjusted = average_price * 1.2
        else:
            return Product.objects.all().values_list('vector_product_id', flat=True)  # Default case when no condition matches
        
        return update_product_uuids(average_condition, average_price_adjusted)
    
def reset_word_tokenize(user_search_prompt=None):
    user_search_prompt = user_search_prompt.replace(" ", "")
    user_search_prompt_tokenize = " ".join(word_tokenize(user_search_prompt, engine="newmm"))

    return user_search_prompt_tokenize
