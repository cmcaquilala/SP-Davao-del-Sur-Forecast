from django.urls import path

from django.conf.urls.static import static
from django.conf import settings

from . import views

urlpatterns = [
    path('', views.indexPage, name='indexPage'),
    path('rice', views.ricePage, name='ricePage'),
    path('corn', views.cornPage, name='cornPage'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)