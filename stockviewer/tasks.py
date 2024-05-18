from celery import shared_task
from .util.stock_retriever import get_multi_stock_quotes
from channels.layers import get_channel_layer
import asyncio
import simplejson as json

@shared_task(bind=True)
def update_quotes(self, stockpicker):
    
    
    selected_stocks = []
    # remove duplicate entries

    for i in list(stockpicker):
        if i is not None or i is not '':
        
            i = i.split(',')
            append_if_not_exists = lambda input_list: [selected_stocks.append(x) for x in input_list if x not in selected_stocks]

            append_if_not_exists(i)

    # print(selected_stocks)
    df = get_multi_stock_quotes(selected_stocks)
    # print(df)

    # send data to group
    channel_layer = get_channel_layer()
    loop = asyncio.new_event_loop()

    asyncio.set_event_loop(loop)

    loop.run_until_complete(channel_layer.group_send("stock_track", {
        "type": 'send_update_quotes',
        'message': df,
        }
    ))

    return 'Done'

    

