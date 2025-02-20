from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("feature1/", views.feature1, name="feature1"),  
    path("feature2/", views.feature2, name="feature2"),  
]
