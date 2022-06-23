__saved_context__ = {}

def savePythonContext():
    import sys
    __saved_context__.update(sys.modules[__name__].__dict__)

def restorePythonContext():
    import sys
    names = list(sys.modules[__name__].__dict__.keys())
    for n in names:
        if n not in __saved_context__:
            del sys.modules[__name__].__dict__[n]

