from django import forms

from basic.models import Invoice, Receipt
from django.forms import widgets


class UploadInvoiceForm(forms.ModelForm):
    class Meta:
        model = Invoice
        fields = ('file', 'deadline_date', 'price', 'name_of_invoicer', 'payed')
        widgets = {
            'deadline_date': widgets.DateTimeInput(attrs={'type': 'date'})
        }

class UploadForm(forms.ModelForm):
    class Meta:
        fields = ('type', 'deadline_date', 'price', 'name_of_invoicer', 'payed')
        widgets = {
            'deadline_date': widgets.DateTimeInput(attrs={'type': 'date'})
        }

class UploadReceiptForm(forms.ModelForm):
    class Meta:
        model = Receipt
        fields = ('file', 'buying_date', 'name_of_article', 'name_of_store', 'price','days_of_warranty')
        widgets = {
            'buying_date': widgets.DateTimeInput(attrs={'type': 'date'})
        }