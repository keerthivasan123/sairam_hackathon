from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponseRedirect
# Create your views here.

def index(request):
    return render(request, 'autism/home.html')

def predict(request):
    return render(request, )