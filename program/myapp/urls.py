from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("feature1/", views.feature1, name="feature1"),  
    path('process-url/', views.process_url, name='process_url'),
    path("feature2/", views.feature2, name="feature2"),  
]
