# Generated by Django 5.0.1 on 2024-05-02 22:22

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('stockportfolio', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='stockfoliouser',
            name='first_name',
            field=models.CharField(default='', max_length=50),
        ),
        migrations.AlterField(
            model_name='stockfoliouser',
            name='last_name',
            field=models.CharField(default='', max_length=50),
        ),
    ]