# Generated by Django 5.0.1 on 2024-05-14 06:41

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('stockportfolio', '0002_alter_stockfoliouser_first_name_and_more'),
        ('stockviewer', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='stockdetails',
            name='user',
            field=models.ManyToManyField(to='stockportfolio.stockfoliouser'),
        ),
    ]
