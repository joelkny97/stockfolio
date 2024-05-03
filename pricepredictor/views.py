from django.shortcuts import render, redirect
from django.urls import reverse
from django.contrib.auth.decorators import login_required

from pricepredictor.predictors.gan_predictor import run_predictor
from pricepredictor.forms import TickerForm
from django.http import HttpResponseRedirect

# Create your views here.


@login_required
def predictor(request):
    if request.method == 'POST':
        form = TickerForm(request.POST)
        if form.is_valid():
            ticker = request.POST['ticker']
            

            return HttpResponseRedirect(reverse('predict', args=[ticker, ]))
    else:
        form = TickerForm()
    return render(request, 'pricepredictor/predictor.html' , {'form': form})

@login_required
def predict(request, symbol):
    prediction_dict = {0:'Hold', 1:'Buy', -1:'Sell'}
    
    context = {}
    
    prediction = run_predictor(symbol)
    context['prediction'] = prediction_dict[prediction.values[0]]
    print(prediction)
    return render(request, 'pricepredictor/predict.html', context)