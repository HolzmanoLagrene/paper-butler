from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('upload_file', views.upload_file, name='upload_file'),
    path('specify_document', views.specify_document, name='specify_document'),
    path('invoices', views.invoice_overview, name='invoices'),
    path('receipts', views.receipts_overview, name='receipts'),
    path('overview', views.document_overview, name='overview'),
    path('invoices_list', views.InvoiceListView.as_view()),
    path('receipts_list', views.ReceiptListView.as_view()),
    path('document_overview', views.DocumentListView.as_view()),

]