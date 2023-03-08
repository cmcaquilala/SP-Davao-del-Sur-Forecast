from django.shortcuts import render
from .utils import get_plot

# Create your views here.
def indexPage(request):


    return render(request, 'dashboard/index.html')