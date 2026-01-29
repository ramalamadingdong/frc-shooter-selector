from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('api/analyze', views.analyze, name='analyze'),
    path('api/estimate-slip', views.estimate_slip, name='estimate_slip'),
]
