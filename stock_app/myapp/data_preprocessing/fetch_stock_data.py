import requests
import json
import numpy as np
import pandas
from datetime import datetime,timedelta
import os
from django.conf import settings

entries = '4000'
start_date =str(datetime.today().date()-timedelta(days=500))
today = str(datetime.today().date()-timedelta(days=0))


def fetch(stock):
    url = f'https://www.nepalipaisa.com/api/GetStockHistory?stockSymbol={stock}&fromDate={start_date}&toDate={today}&pageNo=1&itemsPerPage={entries}&pagePerDisplay=5&_=1725526625076'
    response = requests.get(url)
    jsn = response.json()
    stock_data = jsn.get('result').get('data')
    print(today)
    data = []
    for row in stock_data:
        temp = [
            datetime.strptime(row['tradeDateString'], '%Y-%m-%d').date(),
            float(row['maxPrice']),
            float(row['minPrice']),
            float(row['closingPrice']),
            float(row['noOfTransactions']),
            float(row['volume']),
            float(row['amount']),
            float(row['previousClosing']),
            float(row['differenceRs']),
            float(row['percentChange'])
        ]
        data.append(temp)
    
    head = ['date', 'high', 'low', 'close', 'noOfTransaction', 'volume', 'amount', 'open', 'change', 'chgPercent']
    data.reverse()
    
    df = pandas.DataFrame(data, columns=head)

    if df['date'].iloc[-1]==df['date'].iloc[-2]:
        df=df.drop(df.index[-1])
    if df['noOfTransaction'].iloc[-1]==0.0:
        df=df.drop(df.index[-1])
    df.to_csv(os.path.join(settings.BASE_DIR,'../','data','raw_data',f'{stock}.csv'), index=False)


    return df
