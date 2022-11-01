from django.urls import path

from . import views

urlpatterns = [
    path('upload_invoice', views.upload_invoice, name='upload_invoice'),
    path('upload_receipt', views.upload_receipt, name='upload_receipt'),
    path('invoices', views.invoice_overview, name='invoices'),
    path('receipts', views.receipts_overview, name='receipts'),
    path('invoices_list', views.InvoiceListView.as_view()),
    path('receipts_list', views.ReceiptListView.as_view()),
]