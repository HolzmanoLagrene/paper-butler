from django.urls import path, re_path

from . import views

urlpatterns = [
    path('camera', views.camera, name="camera"),
    path('livecamera', views.livecamera, name="livecamera")
]
