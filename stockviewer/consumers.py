import json
import copy
from channels.generic.websocket import AsyncWebsocketConsumer
from .models import StockDetails 
from  urllib.parse import parse_qs
from django_celery_beat.schedulers import PeriodicTask, IntervalSchedule
from asgiref.sync import async_to_sync, sync_to_async
class StockConsumer(AsyncWebsocketConsumer):

    @sync_to_async
    def add_to_celery_beat(self, stockpicker):
        task = PeriodicTask.objects.filter(name="every-10-seconds")
        if len(task) > 0:
            task = task.first()
            args = json.loads(task.args)
            args = args[0]
            for x  in stockpicker:
                if x not in args:
                    args.append(x)
            task.args = json.dumps([args])
            task.save()
        else:
            schedule, created = IntervalSchedule.objects.get_or_create(every=10, period=IntervalSchedule.SECONDS)
            task = PeriodicTask.objects.create(name="every-10-seconds", task='stockviewer.tasks.update_quotes',interval=schedule, args=json.dumps([stockpicker]))

    @sync_to_async
    def add_to_stockdetail(self, stockpicker):
        user = self.scope["user"]
        for i in stockpicker:
            stock, created  = StockDetails.objects.get_or_create( stock=i)
            stock.user.add(user)
    async def connect(self):
        self.room_name = self.scope["url_route"]["kwargs"]["room_name"]
        self.room_group_name = f"stock_{self.room_name}"

        # Join room group
        await self.channel_layer.group_add(
            self.room_group_name, self.channel_name)

        #parse query_string
        query_params = parse_qs(self.scope["query_string"].decode())

        print(query_params)
        selected_stocks = query_params['selected_stocks']

        # add to celery beat
        await self.add_to_celery_beat(selected_stocks)
        
        #add user to stockdetail
        await self.add_to_stockdetail(selected_stocks)

        await self.accept()
    @sync_to_async
    def helper_func(self):
        user = self.scope["user"]
        stocks = StockDetails.objects.filter(user=user)

        tasks = PeriodicTask.objects.get(name="every-10-seconds")
        args = json.loads(tasks.args)
        args = args[0]
        for i in stocks:
            i.user.remove(user)
            if i.user.count() == 0:
                args.remove(i.stock)
                i.delete()
        if args is None:
            args = []
        if len(args) == 0:
            tasks.delete()
        tasks.args = json.dumps([args])
        tasks.save()
    async def disconnect(self, close_code):

        await self.helper_func()
        # Leave room group
        await self.channel_layer.group_discard(self.room_group_name, self.channel_name)

    # Receive message from WebSocket
    async def receive(self, text_data):
        text_data_json = json.loads(text_data)
        message = text_data_json["message"]

        # Send message to room group
        await self.channel_layer.group_send(
            self.room_group_name, {"type": "send_update", "message": message}
        )
    @sync_to_async
    def select_user_stocks(self):
        user = self.scope["user"]
        user_stocks = StockDetails.objects.filter(user=user).values_list('stock', flat=True)
        # print(user_stocks)
        # user_stocks = user.stockdetails_set.values_list('stock', flat=True).filter(user=user)

        
        return list(user_stocks)
    # Receive message from room group
    async def send_update_quotes(self, event):
        message = event["message"]
        message = copy.copy(message)

        user_stocks = await self.select_user_stocks()
        

        keys = set([i['symbol'] for i in message])
        print(keys)
        for key in list(keys):
            if key in user_stocks:
                pass
            else:
                
                del_idx = [idx-1 for idx,_ in enumerate(message) if _['symbol'] == key ]
                for i in del_idx:
                    del message[i]

        # Send message to WebSocket
        await self.send(text_data=json.dumps(message
        ))