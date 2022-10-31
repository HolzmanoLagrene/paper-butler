import datetime

from django.db import models


class Invoice(models.Model):
    upload_date_time = models.DateTimeField(auto_now_add=True)
    deadline_date = models.DateTimeField()
    price = models.FloatField()
    name_of_invoicer = models.CharField(max_length=100)
    payed = models.BooleanField(default=False)


class Receipt(models.Model):
    upload_date_time = models.DateTimeField(auto_now_add=True)
    buying_date = models.DateTimeField()
    name_of_article = models.CharField(max_length=100)
    name_of_store = models.CharField(max_length=100)
    price = models.FloatField()
    days_of_warranty = models.PositiveBigIntegerField(default=365)

    @property
    def date_of_expiry(self):
        return self.buying_date + datetime.timedelta(days=self.days_of_warranty)
