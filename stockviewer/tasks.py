from celery import shared_task
from .util.stock_retriever import get_multi_stock_quotes
from channels.layers import get_channel_layer
import asyncio
import simplejson as json

@shared_task(bind=True)
def update_quotes(self, stockpicker):
    

    # print(selected_stocks)
    df = get_multi_stock_quotes(stockpicker)

    # send data to group
    channel_layer = get_channel_layer()
    loop = asyncio.new_event_loop()

    asyncio.set_event_loop(loop)

    loop.run_until_complete(channel_layer.group_send("stock_track", {
        "type": 'send_update_quotes', 'message': df},

    ))

    return 'Done'

    

