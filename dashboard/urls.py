from django.urls import path

from django.conf.urls.static import static
from django.conf import settings

from . import views

urlpatterns = [
    path('', views.index_page, name='index_page'),
    path('rice', views.rice_page, name='rice_page'),
    path('rice/add_sarima', views.rice_page_add_sarima, name='rice_page_add_sarima'),
    path('rice/delete_sarima/<int:id>', views.rice_page_delete_sarima, name='rice_page_delete_sarima'),
    path('corn', views.cornPage, name='cornPage'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)