from time import time
import numpy as np


function_times = {}


def time_func(fun, cumulative=True):
    def wrap_fun(*args, **kwargs):
        start_time = time()
        result = fun(*args, **kwargs)
        end_time = time()
        diff_time = end_time - start_time
        function_times.setdefault(fun.__name__, 0.0)
        if cumulative:
            function_times[fun.__name__] += diff_time
        else:
            function_times[fun.__name__] = diff_time
        return result
    return wrap_fun


def time_method(fun, cumulative=True):
    def wrap_method(self, *args, **kwargs):
        start_time = time()
        result = fun(self, *args, **kwargs)
        end_time = time()
        diff_time = end_time - start_time
        identifier = f'{self.__class__.__name__}.{fun.__name__}'
        function_times.setdefault(identifier, 0.0)
        if cumulative:
            function_times[identifier] += diff_time
        else:
            function_times[identifier] = diff_time
        return result
    return wrap_method


def list_timings():
    tmp = [(name, time) for name, time in function_times.items()]
    names = np.array([t[0] for t in tmp])
    times = np.array([t[1] for t in tmp])
    sort_idcs = np.argsort(times)
    names = names[sort_idcs]
    times = times[sort_idcs]
    for n, t in zip(names, times):
        print(f'{n}: {t}')
