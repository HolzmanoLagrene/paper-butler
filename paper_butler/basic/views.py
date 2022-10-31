import datetime

from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse

from basic.models import Invoice


def index(request):
    inv_2 = Invoice.objects.create(name_of_invoicer="test", price=300, deadline_data=datetime.date.today())
    return HttpResponse("Hello, world. You're at the polls index.")