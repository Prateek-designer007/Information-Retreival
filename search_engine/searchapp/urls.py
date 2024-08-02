from django.urls import path 
from .views import results, home

urlpatterns = [
    path('result/', results, name='result'),
    path('',home,name = 'home')
]