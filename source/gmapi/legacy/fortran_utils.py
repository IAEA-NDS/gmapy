from .generic_utils import flatten, static_vars
from fortranformat import FortranRecordReader, FortranRecordWriter


@static_vars(frr_cache={})
def fort_read(fobj, formatstr, none_as=None, debug=False):

    if formatstr not in fort_read.frr_cache:
        fort_read.frr_cache[formatstr] = FortranRecordReader(formatstr) 
    frr = fort_read.frr_cache[formatstr]

    if not isinstance(fobj, str):
        fname = fobj.name
        inpline = fobj.readline()
    else:
        fname = 'console'
        inpline = fobj

    res = frr.read(inpline)
    if none_as is not None:
        res = [none_as if x is None else x for x in res]

    if debug:
        print('--- reading ---')
        print('file: ' + fname)
        print('fmt: ' + formatstr)
        print('str: ' + inpline)

    return res


@static_vars(frw_cache={})
def fort_write(fobj, formatstr, values, retstr=False, debug=False):
    vals = list(flatten(values))
    vals = [v for v in vals if v is not None]
    if debug:
        print('--- writing ---')
        try:
            print('file: ' + fobj.name)
        except AttributeError:
            print('file: console')
        print('fmt: ' + formatstr)
        print('values: ')
        print(vals)

    if formatstr not in fort_write.frw_cache:
        fort_write.frw_cache[formatstr] = FortranRecordWriter(formatstr)
    frw = fort_write.frw_cache[formatstr]

    line = frw.write(vals)
    if fobj is None:
        if retstr:
            return line
        else:
            print(line)
    else:
        fobj.write(line + '\n')


def fort_range(*args):
    if len(args) == 2:
        return range(args[0], args[1]+1)
    elif len(args) == 3:
        return range(args[0], args[1]+1, args[2])
    else:
        raise IndexError

