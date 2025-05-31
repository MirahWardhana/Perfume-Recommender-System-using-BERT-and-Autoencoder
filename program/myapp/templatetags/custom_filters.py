from django import template

register = template.Library()

@register.filter
def get_item(dictionary, key):
    if not dictionary:
        return 0
    # Try direct
    value = dictionary.get(key)
    if value is not None:
        return value
    # Try lower
    value = dictionary.get(key.lower())
    if value is not None:
        return value
    # Try title
    value = dictionary.get(key.title())
    if value is not None:
        return value
    return 0 