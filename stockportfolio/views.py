from django.shortcuts import render
import django.core.exceptions
from django.shortcuts import render
# Takes care of checking if the user is logged in or not
from django.contrib.auth.decorators import login_required
# Yahoo YQL stockretiever to get the stock infos
from stockviewer.util.stock_retriever import get_multi_stock_quotes, get_history_data
# For workbook to create the historical data of a stock
import xlwt
# Http clients to send the attachment file for historical data
from django.http import HttpResponse
# models i.e. database
from .models import StockPortfolio, StockFolioUser
from stockviewer.util.stock_retriever import read_symbols_file
# Need simplejson to pass dictionart STOCK_SYMBOLS to templates
import json
from django.db.models import ObjectDoesNotExist
from django.utils import timezone
# Create your views here.



def portfolio(request):
  '''The main method for all the user functionality'''
  start_date = timezone.now() - timezone.timedelta(days=30)
  end_date = timezone.now()
  user_id = request.user.id
  print(user_id)
  if request.method == 'POST':
    which_form = request.POST.get('which-form', '').strip()
    if which_form == 'find-stock':
      symbol = request.POST.get('stock', '').strip().split(' ')[0].strip().upper()
      if symbol != '':
        portfolio_stock = portfolio_stocks(user_id)
        money = portfolio_stock['money']
        porfolio = portfolio_stock['portfolio_info']
        return render(request, 'stockportfolio/portfolio.html', {'stock':get_multi_stock_quotes([''+symbol]),  'portfolio' : porfolio,  'money' : money, 'portfolio_rows' : plot(user_id, start_date=start_date, end_date=end_date)})
    
    elif which_form == 'buy-stock':
      symbol = request.POST.get('stock-symbol', '').strip()
      StockPortfolio.buy(user_id, symbol, request.POST.get('shares', '').strip(), request.POST.get('cost-per-share', '').strip())
      portfolio_stock = portfolio_stocks(user_id)
      money = portfolio_stock['money']
      porfolio = portfolio_stock['portfolio_info']
      return render(request, 'stockportfolio/portfolio.html', {'stock':get_multi_stock_quotes([''+symbol]),  'portfolio' : porfolio,  'money' : money, 'portfolio_rows' : plot(user_id, start_date=start_date, end_date=end_date)})
    
    elif which_form == 'buy-sell':
      symbol = request.POST.get('stock-symbol', '').strip()
      
      if request.POST.get('buy-stock'):
        StockPortfolio.buy(user_id, symbol, request.POST.get('shares', '').strip(), request.POST.get('cost-per-share', '').strip())
      elif request.POST.get('sell-stock'):
        StockPortfolio.sell(user_id, symbol, request.POST.get('shares', '').strip(), request.POST.get('cost-per-share', '').strip())
      portfolio_stock = portfolio_stocks(user_id)
      money = portfolio_stock['money']
      porfolio = portfolio_stock['portfolio_info']
      return render(request, 'stockportfolio/portfolio.html', {'stock':get_multi_stock_quotes([''+symbol]),  'portfolio' : porfolio,  'money' : money, 'portfolio_rows' : plot(user_id, start_date=start_date, end_date=end_date)})
    
  portfolio_stock = portfolio_stocks(user_id)
  money = portfolio_stock['money']
  porfolio = portfolio_stock['portfolio_info']

  return render(request, 'stockportfolio/portfolio.html', {'portfolio' : porfolio,  'money' : money, 'portfolio_rows' : plot(user_id, start_date=start_date, end_date=end_date)})
    
def portfolio_stocks(user_id):
  '''Returns the list of stocks in a users portfolio'''
  portfolio_info = []
  try:

    stock_list = StockPortfolio.objects.filter(user=user_id)
    user = StockFolioUser.objects.get(pk=user_id)
    money = {'spent': user.spent, 'earnt': user.earnt, 'value': 0, 'profit': '+'}
    if stock_list:
      symbols = [stock.stock for stock in stock_list]
      if len(symbols) == 1:
        stock_data = get_multi_stock_quotes(symbols)
      else:
        stock_data = get_multi_stock_quotes(symbols)
      for stock in stock_data:
        for stock_from_list in stock_list:
          if stock_from_list.stock == stock['symbol']:
            stock['shares'] = stock_from_list.shares
            stock['cost'] = int(stock_from_list.shares) * float(stock['close'])
            money['value'] += float(stock['cost'])
      portfolio_info = [stock_data] if stock_data.__class__ == dict else stock_data
    if float(money['spent']) > (float(money['value']) + float(money['earnt'])):
      money['profit'] = '-'

  except ObjectDoesNotExist:
    money = {'spent': StockFolioUser._meta.get_field('spent').get_default(), 'earnt': StockFolioUser._meta.get_field('earnt').get_default(), 'value': 0, 'profit': '+'}
  except Exception:
    raise
  return {'portfolio_info' : portfolio_info, 'money' : money}

def plot(user_id, start_date, end_date):
  '''Gets Months of historical info on stock and for the graph plots of portfolio'''
  rows = []
  stocks = StockPortfolio.objects.filter(user=user_id)
  if stocks:
    # stock_data = get_history_data(stock.stock, start_date, end_date)
  
    # try:
    #     for val in stock_data:
    #         ans.append((val["datetime"], val["close"], val["open"], val["high"], val["low"], val["volume"]))
    # except KeyError:
        
    data = [list(reversed(get_history_data(stock.stock, start_date=start_date, end_date=end_date))) for stock in stocks]
    days = [day['datetime'] for day in data[0]]
    for idx, day in enumerate(days):
      first = True
      for stock_index, stock in enumerate(data):
        if len(stock) <= idx:
          continue
        if first:
          row = {'Value' : round(float(stock[idx]['close']) * StockPortfolio.objects.filter(stock=stocks[stock_index].stock, user=user_id)[0].shares, 2), 'Date' : day, 'Percent': (float(stock[idx]['open']) - float(stock[idx]['close'])) / float(stock[idx]['close']) * 100, 'Volume': int(stock[idx]['volume']), 'High': float(stock[idx]['high']), 'Low': float(stock[idx]['low']),'Close': float(stock[idx]['close']),  'Open': float(stock[idx]['open'])}
          first = False
        else:
          row['Date'] = day
          row['Value'] += round(float(stock[idx]['close']) * StockPortfolio.objects.filter(stock=stocks[stock_index].stock, user=user_id)[0].shares, 2)
          row['Volume'] += int(stock[idx]['volume'])
          row['Open'] = (row['Open']  + float(stock[idx]['open']))/2
          row['High'] = (row['High']  + float(stock[idx]['high']))/2
          row['Low'] = (row['Low']  + float(stock[idx]['low']))/2
          row['Close'] = (row['Close']  + float(stock[idx]['close']))/2
          
          row['Percent'] += (float(stock[idx]['open']) - float(stock[idx]['close'])) / float(stock[idx]['close']) * 100
      rows.append(row)
    rows.reverse()
    
  return rows