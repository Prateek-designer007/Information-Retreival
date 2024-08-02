from django.urls import path 
from .views import cluster_result, cluster_home

urlpatterns = [
    path('cluster/result/', cluster_result, name='cluster-result'),
    path('cluster/',cluster_home,name = 'cluster-home'),
]