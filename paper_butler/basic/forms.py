from django import forms
from django.core.files import File

from basic.models import UploadFile, UploadedDocument
from django.forms import widgets


class UploadFileForm(forms.ModelForm):
    store_original = forms.BooleanField()

    class Meta:
        model = UploadFile
        fields = ('file',)


class DocumentSpecificationForm(forms.ModelForm):
    class Meta:
        model = UploadedDocument
        exclude = []
        widgets = {
            'buying_date': widgets.DateTimeInput(attrs={'type': 'date'}),
            'deadline_date': widgets.DateTimeInput(attrs={'type': 'date'}),
            'file_obj': forms.NumberInput(attrs={'class': 'form-control', 'readonly': True}),
        }
