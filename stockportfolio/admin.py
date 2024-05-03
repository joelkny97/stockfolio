from django.contrib import admin

# Register your models here.
from stockportfolio.models import StockFolioUser, StockPortfolio

class StockFolioUserAdmin(admin.ModelAdmin):
    list_display = ('pk','first_name', 'last_name', 'earnt', 'spent')

class StockPortfolioAdmin(admin.ModelAdmin):
    list_display = ('pk','stock', 'shares', 'user')


admin.site.register(StockFolioUser, StockFolioUserAdmin)
admin.site.register(StockPortfolio, StockPortfolioAdmin)