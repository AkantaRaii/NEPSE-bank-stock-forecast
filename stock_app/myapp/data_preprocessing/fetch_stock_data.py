import requests
import json
import numpy as np
import pandas
from datetime import datetime,timedelta


entries = '4000'
start_date =str(datetime.today().date()-timedelta(days=100))
today = str(datetime.today().date()-timedelta(days=1))


def fetch(stock):
    url = f'https://www.nepalipaisa.com/api/GetStockHistory?stockSymbol={stock}&fromDate={start_date}&toDate={today}&pageNo=1&itemsPerPage={entries}&pagePerDisplay=5&_=1725526625076'
    response = requests.get(url)
    jsn = response.json()
    stock_data = jsn.get('result').get('data')
    
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
    print(f'\nlast date in fetching {df['date'].iloc[-1]}and TXN is{df['noOfTransaction'].iloc[-1]}\n')
    print(f'\nlast date in fetching {df['date'].iloc[-2]}and TXN is{df['noOfTransaction'].iloc[-2]}\n')

    if df['date'].iloc[-1]==df['date'].iloc[-2]:
        df=df.drop(df.index[-1])
    if df['noOfTransaction'].iloc[-1]==0.0:
        df=df.drop(df.index[-1])
    print(f'\nlast date in fetching {df['date'].iloc[-1]}and TXN is{df['noOfTransaction'].iloc[-1]}\n')
    print(f'\nlast date in fetching {df['date'].iloc[-2]}and TXN is{df['noOfTransaction'].iloc[-2]}\n')
    return df
fetch('EBL')
