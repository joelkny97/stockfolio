from django import template

register = template.Library()

@register.filter
def get(map, key):
    return map.get(key, '')