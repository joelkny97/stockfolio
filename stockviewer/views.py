from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
# Takes care of all the basic login, authenticizing and login process for tokenization
from django.contrib.auth import login, logout, authenticate
# The in-built django user module will take care of all the user related database functionality
from django.contrib.auth.models import User
# User of this App
from django.urls import reverse
from django.http import HttpResponseRedirect, HttpResponse
# Create your views here.
from .util.stock_retriever import read_symbols_file, get_multi_stock_quotes
from .forms import StockPickerForm
from stockportfolio.models import StockFolioUser


def index(request):
    '''The home page for StockGenFolio'''
    context={}

    # write_symbols_file()
    # symbols = read_symbols_file()
    # print(symbols)



    return render(request, 'stockviewer/index.html')

def stockpicker(request):
    context={}
    #    stocks = read_symbols_file()
    #    context['stock_symbols'] = stocks.keys()
        
    

    if request.method == 'POST':
        form = StockPickerForm(request.POST)
        if form.is_valid():
            
            selected_stocks = form.cleaned_data.get('stockpicker')
            print(selected_stocks)
        return HttpResponseRedirect(reverse('stocktracker') )
    else:
        form = StockPickerForm()

    context['form']=form

    return render(request, 'stockviewer/stockpicker.html',context=context)

def stocktracker(request):
    context={}
    selected_stocks = request.POST.getlist('available_stocks')

    # print(selected_stocks)
    df = get_multi_stock_quotes(selected_stocks)

    context['df'] = df
    context['room_name'] = 'track'
    context['selected_stocks'] = selected_stocks

    # stock_quote_table= [ {'Symbol': i['symbol'], 'Name': i['name'], 'Price': i['price']} i for i in df ] 

    return render(request, 'stockviewer/stocktracker.html', context=context)


def login_user(request):
  '''Form support for Login'''
  if request.method == "POST":
    email = request.POST.get('email', '')
    password = request.POST.get('password', '')

    authenticated = authenticate(username=email, password=password)

    # Valid username and password
    if authenticated is not None and authenticated.is_active:
      login(request, authenticated)
      print("Login Successful")
      return redirect( 'home')
    # Incorrect username or password
    else:
      return render(request, 'stockviewer/login.html', {'errors': ['Invalid Username and/or Password']})

  else:
    # Displacy Login Form
    return render(request, 'stockviewer/login.html')



def register_user(request):
  '''Form support for User Registration Process'''
  if request.method == "POST":
    # Get POST params from request
    f_name = request.POST.get('first_name', '')
    l_name = request.POST.get('last_name', '')
    email = request.POST.get('email', '')
    password = request.POST.get('password', '')
    confirm_password = request.POST.get('password_confirmation', '')

    errors = []
    # Check if the user already exists
    try:
      User.objects.get(username=email)
      errors.append('A user by that username already exists')
    except User.DoesNotExist:
      pass

    # Input Field checks
    if len(password) < 3:
      errors.append('Enter a valid password that is more than 3 characters')
    if password != confirm_password:
      errors.append('Password and Confirm Password don\'t match')
    if len(errors) > 0:
      return render(request, 'stockviewer/register.html', {'errors' : errors})

    # Create a User and redirect to login
    user = User.objects.create_user(email, password=password)
    StockFolioUser.objects.create(user=user, first_name=f_name, last_name=l_name)
    return render(request, 'stockviewer/login.html')
  else:
    # Display registration form 
    return render(request, 'stockviewer/register.html')

def logout_user(request):
  '''Logouts the currently signed in user and redirects to login'''
  logout(request)
  return redirect(reverse('login'))


def about(request):
    return render(request, 'stockviewer/about.html')

def team(request):
    return render(request, 'stockviewer/team.html')

