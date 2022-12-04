import datetime

from django.shortcuts import render, redirect
from django_serverside_datatable.views import ServerSideDatatableView

# Create your views here.
from django.http import HttpResponse, HttpResponseRedirect

from basic.forms import UploadFileForm, DocumentSpecificationForm
from basic.models import Invoice, Receipt, UploadFile, DocumentType, UploadedDocument
from image_processing.image_handler import ImageHandler
from image_processing.utils import FileType
from utils.utils import get_human_readable_id


class InvoiceListView(ServerSideDatatableView):
    queryset = Invoice.objects.all()
    columns = ['upload_date_time', 'deadline_date', 'price', "name_of_invoicer", "payed"]


class ReceiptListView(ServerSideDatatableView):
    queryset = Receipt.objects.all()
    columns = ['upload_date_time', 'buying_date', "name_of_article", "name_of_store", "price",
               "days_of_warranty"]

def document_overview(request):
    return render(request, "basic/overview.html")

def receipts_overview(request):
    return render(request, "basic/receipts.html")


def invoice_overview(request):
    return render(request, "basic/invoices.html")


def upload_file(request):
    if request.method == "POST":
        upload_file_form = UploadFileForm(request.POST, request.FILES)
        if upload_file_form.is_valid():
            if upload_file_form.cleaned_data["store_original"]:
                human_readable_id = get_human_readable_id()
            upload_file_tmp_obj = upload_file_form.save()
            file_type = ImageHandler.classify_image(upload_file_tmp_obj)
            uploaded_document = UploadedDocument(file_obj=upload_file_tmp_obj, type=file_type, human_readable_id=human_readable_id)
            uploaded_document.save()
            form = DocumentSpecificationForm(instance=uploaded_document)
        else:
            specify_document_form = DocumentSpecificationForm(request.POST)
            if specify_document_form.is_valid():
                specify_document_form.save()
            return render(request, 'basic/index.html')
    else:
        form = UploadFileForm
    return render(request, 'basic/upload_document.html', {'form': form})


def specify_document(request):
    if request.method == "POST":
        form = DocumentSpecificationForm(request.POST)
        if form.is_valid():
            form.save()
        return redirect("invoices")
    else:
        form = DocumentSpecificationForm
        return render(request, 'basic/upload_document.html', {'form': form})


def index(request):
    return render(request, "basic/index.html")


class DocumentListView(ServerSideDatatableView):
    queryset = UploadedDocument.objects.all()
    columns = ['upload_date_time', 'type','deadline_date', 'price', "name_of_invoicer", "payed"]
