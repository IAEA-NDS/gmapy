import os
import dill


def load_objects(filename, *varnames):
    with open(filename, 'rb') as f:
        objdic = dill.load(f)
    return tuple(objdic[v] for v in varnames)


def save_objects(filename, scope, *varnames):
    dirname = os.path.dirname(filename)
    os.makedirs(dirname, exist_ok=True)
    objdic = {v: scope[v] for v in varnames}
    with open(filename, 'wb') as f:
        dill.dump(objdic, f)
