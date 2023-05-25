from django.urls import path

from django.conf.urls.static import static
from django.conf import settings

from . import views
from .views_x import *

urlpatterns = [
    path('', views.index_page, name='index_page'),
    path('<str:dataset>', views.graphs_page, name='graphs_page'),
    path('add_sarima/<str:dataset>', views.add_sarima, name='add_sarima'),
    path('add_bayesian/<str:dataset>', views.add_bayesian, name='add_bayesian'),
    path('delete_sarima/<str:dataset>/<int:id>', views.delete_sarima, name='delete_sarima'),
    path('delete_bayesian/<str:dataset>/<int:id>', views.delete_bayesian, name='delete_bayesian'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)