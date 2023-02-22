from time import time


def time_func(fun):
    def wrap_fun(*args, **kwargs):
        start_time = time()
        result = fun(*args, **kwargs)
        end_time = time()
        diff_time = end_time - start_time
        print(f'Function {fun.__name__!r} executed in '
              f'{diff_time:.4f}s')
        return result
    return wrap_fun


def time_method(fun):
    def wrap_method(self, *args, **kwargs):
        start_time = time()
        result = fun(self, *args, **kwargs)
        end_time = time()
        diff_time = end_time - start_time
        print(f'Function {self.__class__.__name__}.{fun.__name__!r} '
              f'executed in {diff_time:.4f}s')
        return result
    return wrap_method
