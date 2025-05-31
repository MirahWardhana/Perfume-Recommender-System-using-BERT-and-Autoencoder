from django import template

register = template.Library()

@register.filter
def get_item(dictionary, key):
    if not dictionary:
        return 0
    # Try to get the value directly
    value = dictionary.get(key)
    if value is not None:
        return value
    # If not found, try with the key in lowercase
    value = dictionary.get(key.lower())
    if value is not None:
        return value
    # If still not found, try with the key in title case
    value = dictionary.get(key.title())
    if value is not None:
        return value
    return 0 