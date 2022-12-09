from django import forms

from basic.models import Document
from django.forms import widgets


class UploadDocumentForm(forms.ModelForm):
    store_original = forms.BooleanField(required=False)

    class Meta:
        model = Document
        fields = ('file_obj', 'store_original',)


class SpecifyDocumentForm(forms.ModelForm):
    class Meta:
        model = Document
        exclude = ('file_obj', 'human_readable_id',)
        widgets = {
            'buying_date': widgets.DateTimeInput(attrs={'type': 'date'}),
            'deadline_date': widgets.DateTimeInput(attrs={'type': 'date'})
        }


class SpecifyDocumentAsInvoiceForm(forms.ModelForm):
    class Meta:
        model = Document
        exclude = ('file_obj',)
        widgets = {
            'buying_date': widgets.DateTimeInput(attrs={'type': 'date'}),
            'deadline_date': widgets.DateTimeInput(attrs={'type': 'date'})
        }


class SpecifyDocumentAsReceiptForm(forms.ModelForm):
    class Meta:
        model = Document
        exclude = ('file_obj',)
        widgets = {
            'buying_date': widgets.DateTimeInput(attrs={'type': 'date'}),
            'deadline_date': widgets.DateTimeInput(attrs={'type': 'date'})
        }
