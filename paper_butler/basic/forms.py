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
        exclude = ('file_obj', 'human_readable_id', 'used_for_training',)
        widgets = {
            'buying_date': widgets.DateTimeInput(attrs={'type': 'date'}),
            'deadline_date': widgets.DateTimeInput(attrs={'type': 'date'})
        }


class SpecifyDocumentAsUnknownForm(forms.ModelForm):
    class Meta:
        model = Document
        fields = ('type',)


class SpecifyDocumentAsInvoiceForm(forms.ModelForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for field in self.Meta.required:
            self.fields[field].required = True

    class Meta:
        model = Document
        exclude = (
            'file_obj', 'buying_date', 'name_of_article', 'name_of_store', 'human_readable_id', 'used_for_training',)
        required = (
            'deadline_date',
            'name_of_invoicer',
            'payed',
            'price',
        )
        widgets = {
            'deadline_date': widgets.DateTimeInput(attrs={'type': 'date'})
        }


class SpecifyDocumentAsReceiptForm(forms.ModelForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for field in self.Meta.required:
            self.fields[field].required = True

    class Meta:
        model = Document
        exclude = ('file_obj', 'deadline_date', 'name_of_invoicer', 'payed', 'human_readable_id', 'used_for_training',)
        required = (
            'buying_date',
            'name_of_article',
            'name_of_store',
            'days_of_warranty',
        )
        widgets = {
            'buying_date': widgets.DateTimeInput(attrs={'type': 'date'}),
        }
