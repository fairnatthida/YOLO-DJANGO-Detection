from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),  # URL path for the home page (index)
    path('detect/', views.detect, name='detect'),  # URL path for object detection (detect)
]
