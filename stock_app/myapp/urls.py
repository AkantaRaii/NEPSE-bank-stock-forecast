from django.urls import path
from . import views
urlpatterns=[
    path('',views.home,name='home'),
    path('stock/<str:stock_name>',views.stock,name='stock'),
]