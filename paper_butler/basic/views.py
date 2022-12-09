from django.shortcuts import render, redirect
from django_serverside_datatable.views import ServerSideDatatableView

import utils.utils
from basic.forms import UploadDocumentForm, SpecifyDocumentForm
from basic.models import Document
from image_processing.image_handler import ImageHandler
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
    columns = ['upload_date_time', 'type', 'physical_copy_exists']


def change_document(request, id):
    if utils.utils.is_human_readable_id(id):
        document = Document.objects.filter(human_readable_id=id).get()
    else:
        document = Document.objects.filter(id=id).get()
    if request.method == "POST":
        form = SpecifyDocumentForm(request.POST, instance=document)
        if form.is_valid():
            form.save()
        else:
            print()
        return redirect('/')
    else:
        form = SpecifyDocumentForm(instance=document)
        if document.human_readable_id:
            id = document.human_readable_id
            has_hr_id = True
        else:
            id = document.id
            has_hr_id = False
        return render(request, 'basic/change_document.html', {'form': form, 'id': id, 'has_hr_id': has_hr_id})


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
            print()
        return redirect(f"change_document/{id}")
    else:
        form = UploadDocumentForm
        return render(request, 'basic/upload_document.html', {'form': form})
