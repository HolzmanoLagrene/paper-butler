from django.urls import path, re_path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('change_document/<str:id>', views.change_document, name="change_document"),
    path('upload_document', views.upload_document, name='upload_document'),
    path('invoices', views.invoice_overview, name='invoices'),
    path('receipts', views.receipts_overview, name='receipts'),
    path('overview', views.document_overview, name='overview'),
    path('document_overview', views.DocumentListView.as_view()),
    path('train', views.train,name="train")

]
