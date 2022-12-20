import datetime
from django.utils.translation import gettext_lazy as _

from django.db import models


class DocumentType(models.TextChoices):
    Unknown = 'UK', _('Unknown')
    Invoice = 'IV', _('Invoice')
    Receipt = 'RC', _('Receipt')


class Document(models.Model):
    upload_date_time = models.DateTimeField(auto_now_add=True)
    file_obj = models.FileField(upload_to='original', null=False, blank=False)
    type = models.CharField(max_length=2, choices=DocumentType.choices, default=DocumentType.Unknown, blank=False, null=False)
    deadline_date = models.DateTimeField(blank=True, null=True)
    name_of_invoicer = models.CharField(max_length=100, blank=True, null=True)
    payed = models.BooleanField(default=False, blank=True, null=True)
    buying_date = models.DateTimeField(blank=True, null=True)
    name_of_article = models.CharField(max_length=100, blank=True, null=True)
    name_of_store = models.CharField(max_length=100, blank=True, null=True)
    price = models.FloatField(blank=True, null=True)
    days_of_warranty = models.PositiveBigIntegerField(default=365, blank=True, null=True)
    human_readable_id = models.CharField(max_length=100, blank=True, null=True)
    used_for_training = models.BooleanField(default=False, blank=True, null=False)

    @property
    def date_of_expiry(self):
        return self.buying_date + datetime.timedelta(days=self.days_of_warranty)

    @property
    def physical_copy_exists(self):
        if self.human_readable_id:
            return True
        else:
            return False
