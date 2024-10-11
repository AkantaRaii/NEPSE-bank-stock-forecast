from django.shortcuts import render
from .model.inference import predict

# Create your views here.
def home(request):
    return render(request,'home.html')
def stock(request,stock_name):
    stock_sequennce={'ADBL':3,'EBL':3,'NABIL':20,'SANIMA':3,'NIFRA':10}
    predicted_value=predict(stock_name,stock_sequennce[stock_name])
    return render(request,'stock.html',{'stock_name':stock_name,'predicted_value':predicted_value})