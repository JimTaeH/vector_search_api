import pandas as pd
import numpy as np
from .models import Product
from tqdm import tqdm

def run():
    csv_fpath = "productData/lazada_womenwear_clean2_nodup.csv"
    df = pd.read_csv(csv_fpath)
    df = df[["productName", "productDes", "link", 
             "price", "image", "sold_units", 
             "rating", "no_review", "shipmentOrigin", 
             "brand"]].copy()
    
    df = df.dropna(subset="sold_units").reset_index(drop=True)

    for i in tqdm(range(df.shape[0])):
        productName = df["productName"].iloc[i]
        productDes = df["productDes"].iloc[i]
        productLink = df["link"].iloc[i]
        productPrice = df["price"].iloc[i]
        productImage = df["image"].iloc[i]
        productSoldUnits = df["sold_units"].iloc[i]
        productRating = df["rating"].iloc[i]
        productNoReview = df["no_review"].iloc[i]
        productShipmentOrigin = df["shipmentOrigin"].iloc[i]
        productBrand = df["brand"].iloc[i]

        product = Product(
            productName = productName,
            productDes = productDes,
            image = productImage,
            price = productPrice,
            sold_units = productSoldUnits,
            rating = productRating,
            no_review = productNoReview,
            link = productLink,
            shipmentOrigin = productShipmentOrigin,
            brand = productBrand
        )

        product.save()

    print("Saved to Databases!")