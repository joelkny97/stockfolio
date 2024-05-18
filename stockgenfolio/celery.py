from __future__ import absolute_import, unicode_literals
import os
from celery import Celery
from django.conf import settings
# from celery.schedules import crontab

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'stockgenfolio.settings')

app = Celery('stockgenfolio')
app.conf.enable_utc = False
app.conf.update(timezone = 'US/Eastern')

app.config_from_object('django.conf:settings', namespace='CELERY')

app.conf.beat_schedule = {
    # 'every-30-seconds': {
    #     'task': 'stockviewer.tasks.update_quotes',
    #     'schedule': 30,
    #     'args': ([''],) 
    # },
    
}

app.autodiscover_tasks()

@app.task(bind=True)
def debug_task(self):
    print(f'Request: {self.request!r}')