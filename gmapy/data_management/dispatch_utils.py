def get_method(item, method, type_getter, dispatch_map):
    itemtype = type_getter(item)
    if callable(dispatch_map):
        module = dispatch_map(itemtype)
    else:
        module = dispatch_map.get(itemtype, None)
    if module is None:
        raise ValueError(f'unknown type `{itemtype}`')
    special_method = getattr(module, method)
    return special_method


def call_method(
    item, method, type_getter, dispatch_map, *args, **kwargs
):
    args = [] if args is None else args
    kwargs = {} if kwargs is None else kwargs
    fun = get_method(item, method, type_getter, dispatch_map)
    return fun(item, *args, **kwargs)


def generate_method_caller(type_getter, dispatch_map):
    def customized_call_method(item, method, *args, **kwargs):
        return call_method(
            item, method, type_getter, dispatch_map, *args, **kwargs
        )
    return customized_call_method


def generate_method_getter(type_getter, dispatch_map):
    def customized_get_method(item, method):
        return get_method(item, method, type_getter, dispatch_map)
    return customized_get_method
