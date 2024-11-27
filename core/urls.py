from django.urls import path
from . import views

urlpatterns = [
    path("", views.upload_prescription, name="upload_image"),  # Ensure this maps to an actual view in views.py
]
