import json

from django.contrib import messages
from django.shortcuts import render, redirect
from django_serverside_datatable.views import ServerSideDatatableView
from django.http import JsonResponse

import utils.utils
from basic.forms import UploadDocumentForm, SpecifyDocumentForm, SpecifyDocumentAsUnknownForm, \
    SpecifyDocumentAsReceiptForm, SpecifyDocumentAsInvoiceForm
from basic.models import Document, DocumentType
from image_processing.image_handler import ImageHandler
from image_processing.setup_ml_data import run_training_cylce
from utils.utils import get_human_readable_id


def index(request):
    return render(request, "basic/index.html")


def document_overview(request):
    return render(request, "basic/overview.html")


def receipts_overview(request):
    return render(request, "basic/receipts.html")


def invoice_overview(request):
    return render(request, "basic/invoices.html")


class DocumentListView(ServerSideDatatableView):
    queryset = Document.objects.all()
    columns = ['upload_date_time', 'type']
    # columns = ['upload_date_time', 'type', 'physical_copy_exists']


def change_document(request, id):
    if utils.utils.is_human_readable_id(id):
        try:
            document = Document.objects.filter(human_readable_id=id).get()
            has_human_readable_id = True
            form_id = document.human_readable_id
        except Exception as e:
            messages.error(request, f"There is no document with id {id}")
            return redirect("/")
    else:
        try:
            document = Document.objects.filter(id=id).get()
            has_human_readable_id = False
            form_id = document.id
        except Exception as e:
            messages.error(request, f"There is no document with id {id}")
            return redirect("/")
    if request.method == "PUT":
        type = json.loads(request.body.decode("utf8"))["type"]
        document.type = type
        document.save()
        return JsonResponse({'success': True})
    if request.method == "POST":
        form = SpecifyDocumentForm(request.POST, instance=document)
        if form.is_valid():
            form.save()
        else:
            messages.error(request, f"There was an error with the document")
        return redirect('/')
    else:
        if document.type == DocumentType.Unknown:
            form = SpecifyDocumentAsUnknownForm(instance=document)
        elif document.type == DocumentType.Receipt:
            form = SpecifyDocumentAsReceiptForm(instance=document)
        else:
            form = SpecifyDocumentAsInvoiceForm(instance=document)
        return render(request, 'basic/change_document.html',
                      {'form': form, 'id': form_id, 'has_hr_id': has_human_readable_id})


def upload_document(request):
    if request.method == "POST":
        form = UploadDocumentForm(request.POST, request.FILES)
        if form.is_valid():
            document = form.save(commit=False)
            document.type = ImageHandler.classify_image(document.file_obj)
            document.save()
            if form.cleaned_data["store_original"]:
                id = get_human_readable_id()
                document.human_readable_id = id
            else:
                id = document.id
            document.save()
        else:
            messages.error(request, f"There was an error with thid document")
            return redirect("/")
        return redirect(f"change_document/{id}")
    else:
        form = UploadDocumentForm
        return render(request, 'basic/upload_document.html', {'form': form})


def train(request):
    run_training_cylce()
    return redirect("/")
