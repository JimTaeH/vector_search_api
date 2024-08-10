from django.db import models
import uuid
from pgvector.django import VectorField, HnswIndex

# Create your models here.
class TestProducts(models.Model):
    productID = models.UUIDField(primary_key=True, unique=True, default=uuid.uuid4, editable=False)
    productName = models.CharField(max_length=255)
    productDesc = models.TextField(max_length=5000)
    productLink = models.URLField(max_length=2048)
    productPrice = models.FloatField()

class TestEmbeddings(models.Model):
    product = models.ForeignKey(TestProducts, on_delete=models.CASCADE)
    embedding_name = VectorField(dimensions=384, null=True, blank=True)
    embedding_desc = VectorField(dimensions=384, null=True, blank=True)
    
    class Meta:
        indexes = [
            HnswIndex(
                name='desc_index',
                fields=['embedding_desc'],
                m=16,
                ef_construction=64,
                opclasses=['vector_l2_ops']
            )
        ]

class Product(models.Model): 
    vector_product_id = models.UUIDField(primary_key=True, unique=True, default=uuid.uuid4, editable=False)
    productName = models.CharField(max_length=255)
    productDes = models.TextField(max_length=5000)
    image = models.URLField(max_length=2048)
    price = models.FloatField()
    sold_units = models.IntegerField()
    rating = models.FloatField()
    no_review = models.IntegerField()
    link = models.URLField(max_length=2048) 
    shipmentOrigin = models.CharField(max_length=255)
    brand = models.CharField(max_length=255)

class NameEmbeddings(models.Model):
    product = models.ForeignKey(Product, models.DO_NOTHING)
    embedding_name = VectorField(dimensions=384, null=True, blank=True)

    class Meta:
        indexes = [
            HnswIndex(
                name='pname_index',
                fields=['embedding_name'],
                m=16,
                ef_construction=64,
                opclasses=['vector_l2_ops']
            )
        ]

class DescEmbeddings(models.Model):
    product = models.ForeignKey(Product, models.DO_NOTHING)
    embedding_desc = VectorField(dimensions=384, null=True, blank=True)
    document = models.CharField(blank=True, null=True, max_length=1000)

    class Meta:
        indexes = [
            HnswIndex(
                name='pdesc_index',
                fields=['embedding_desc'],
                m=16,
                ef_construction=64,
                opclasses=['vector_l2_ops']
            )
        ]

