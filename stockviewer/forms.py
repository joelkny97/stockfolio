from django import forms
from stockviewer.util.stock_retriever import read_symbols_file
class StockPickerForm(forms.Form):

    OPTIONS = read_symbols_file()
    # print(OPTIONS)

    available_stocks = forms.MultipleChoiceField(choices=OPTIONS, 
                                                )
    
    
