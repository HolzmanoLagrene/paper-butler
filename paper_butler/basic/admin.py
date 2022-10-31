from django.contrib import admin

from .models import Receipt,Invoice

admin.site.register(Receipt)
admin.site.register(Invoice)