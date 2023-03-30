from django.urls import path

from django.conf.urls.static import static
from django.conf import settings

from . import views

urlpatterns = [
    path('', views.index_page, name='index_page'),
    path('rice', views.rice_page, name='rice_page'),
    # path('add_sarima/rice', views.rice_page_add_sarima, name='rice_page_add_sarima'),
    # path('delete_sarima/rice/<int:id>', views.rice_page_delete_sarima, name='rice_page_delete_sarima'),
    # path('add_bayesian/rice', views.rice_page_add_bayesian, name='rice_page_add_bayesian'),
    # path('delete_bayesian/rice/<int:id>', views.rice_page_delete_bayesian, name='rice_page_delete_bayesian'),

    path('corn', views.corn_page, name='corn_page'),
    # path('add_sarima/corn', views.corn_page_add_sarima, name='corn_page_add_sarima'),
    # path('delete_sarima/corn/<int:id>', views.corn_page_delete_sarima, name='corn_page_delete_sarima'),
    # path('add_bayesian/corn', views.corn_page_add_bayesian, name='corn_page_add_bayesian'),
    # path('delete_bayesian/corn/<int:id>', views.corn_page_delete_bayesian, name='corn_page_delete_bayesian'),

    path('add_sarima/<str:dataset>', views.add_sarima, name='add_sarima'),
    path('add_bayesian/<str:dataset>', views.add_bayesian, name='add_bayesian'),
    path('delete_sarima/<str:dataset>/<int:id>', views.delete_sarima, name='delete_sarima'),
    path('delete_bayesian/<str:dataset>/<int:id>', views.delete_bayesian, name='delete_bayesian'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)