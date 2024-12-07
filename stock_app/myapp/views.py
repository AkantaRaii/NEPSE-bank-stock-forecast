from django.shortcuts import render
from .model.inference import Predict
from datetime import datetime,timedelta
from .data import Data
    

# Create your views here.
def home(request):
    return render(request,'home.html')
def stock(request,stock_name):
    p1=Predict()
    data=Data()
    lasso_predicted_value,date=p1.predict_lasso(stock_name,data.lasso_seq[stock_name])
    pca_predicted_value,date=p1.predict_pca(stock_name,data.pca_seq[stock_name])
    print(lasso_predicted_value,pca_predicted_value)
    return render(request,'stock.html',{'stock_name':stock_name,'lasso_predicted_value':(round(lasso_predicted_value[-1],2)),'pca_predicted_value':(round(pca_predicted_value[-1],2)),'date':str(date[-1]+timedelta(days=1))})