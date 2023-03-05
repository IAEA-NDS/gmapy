import json
from ..legacy.database_reading import read_gma_database as read_legacy_gma_database_basic
from ..legacy.conversion_utils import sanitize_datablock, sanitize_prior


def read_legacy_gma_database(dbfile):
    format_dic = {}
    db_dic = read_legacy_gma_database_basic(dbfile, format_dic=format_dic)
    legacy_datablock_list = db_dic['datablock_list']
    prior_list = sanitize_prior(db_dic['APR'])
    datablock_list = [sanitize_datablock(b) for b in legacy_datablock_list]
    return {'prior_list': prior_list, 'datablock_list': datablock_list}


def read_json_gma_database(dbfile):
    with open(dbfile, 'r') as f:
        db_dic = json.load(f)
    return {'prior_list': db_dic['prior'],
            'datablock_list': db_dic['datablocks']}


def read_gma_database(dbfile):
    with open(dbfile, 'r') as f:
        firstline = f.readline()
    is_json = firstline.strip().startswith('{')
    if is_json:
        db = read_json_gma_database(dbfile)
    else:
        db = read_legacy_gma_database(dbfile)
    return db
