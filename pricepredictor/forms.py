from typing import Any
from django import forms


class TickerForm(forms.Form):
    ticker = forms.CharField(max_length=5, label='Ticker Symbol',widget=forms.TextInput(attrs={'class': "form-select fw-bold",'style': 'max-width: auto; width: 100%;'}))
    # start_date = forms.DateField(label='Start Date',widget=forms.widgets.DateInput(attrs={'type': 'date','class': "form-select text-center fw-bold",'style': 'max-width: auto; justify-content: center; width: 100%;'}))
    # end_date = forms.DateField(label='End Date',widget=forms.widgets.DateInput(attrs={'type': 'date','class': "form-select text-center fw-bold",'style': 'max-width: auto; justify-content: center; width: 100%;'}))

    def clean(self) -> dict[str, Any]:
        cleaned_data = super().clean()
        # clean_start_date = cleaned_data.get('start_date')
        # clean_end_date = cleaned_data.get('end_date')
        # if clean_start_date > clean_end_date:
        #     raise forms.ValidationError('Start date cannot be greater than end date')
        return cleaned_data