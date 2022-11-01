import datetime

from django.shortcuts import render, redirect
from django_serverside_datatable.views import ServerSideDatatableView

# Create your views here.
from django.http import HttpResponse, HttpResponseRedirect

from basic.forms import UploadInvoiceForm, UploadReceiptForm
from basic.models import Invoice, Receipt


class InvoiceListView(ServerSideDatatableView):
    queryset = Invoice.objects.all()
    columns = ['upload_date_time', 'deadline_date', 'price', "name_of_invoicer", "payed"]


class ReceiptListView(ServerSideDatatableView):
    queryset = Receipt.objects.all()
    columns = ['upload_date_time', 'buying_date', "name_of_article", "name_of_store", "price",
               "days_of_warranty"]


def receipts_overview(request):
    return render(request, "basic/receipts.html")


def invoice_overview(request):
    return render(request, "basic/invoices.html")


def handle_uploaded_file(param):
    pass


def upload_invoice(request):
    if request.method == "POST":
        form = UploadInvoiceForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
        return redirect("invoices")
    else:
        form = UploadInvoiceForm
    return render(request, 'basic/upload.html', {'form': form})

def upload_receipt(request):
    if request.method == "POST":
        form = UploadReceiptForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
        return receipts_overview(request)
    else:
        form = UploadReceiptForm
    return render(request, 'basic/upload.html', {'form': form})
