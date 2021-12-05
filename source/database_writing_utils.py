from fortran_utils import fort_write
from gmap_snippets import get_dataset_range, get_prior_range, MT_to_label
from database_reading import read_gma_database

import numpy as np
import pandas as pd


def write_control_header(ACON, MC1, MC2, MC3, MC4, MC5, MC6, MC7, MC8):
    format100 = "(A4,1X,8I5)"
    line = fort_write(None, format100,
            [ACON, MC1, MC2, MC3, MC4, MC5, MC6, MC7, MC8], retstr=True)
    return line



def write_IPP_setup(IPP1, IPP2, IPP3, IPP4, IPP5, IPP6, IPP7, IPP8):
    line = write_control_header('I/OC',
            IPP1, IPP2, IPP3, IPP4, IPP5, IPP6, IPP7, IPP8)
    return line



def write_mode_setup(MODC, MOD2, AMO3, MODAP, MPPP):
    line = write_control_header('MODE', MODC, MOD2, AMO3, MODAP, MPPP, 0, 0, 0)
    return line



def generate_dataset_text(dataset, format_dic={}):   
    # error checks
    if len(dataset['reference']) > 32:
        raise ValueError('reference must not be longer than 32 characters')

    if len(dataset['author']) > 32:
        raise ValueError('author field must not be longer than 32 characters')

    if not isinstance(dataset['dataset_id'], int):
        raise ValueError('dataset_id must be integer')

    if not isinstance(dataset['year'], int):
        raise ValueError('year must be integer')

    if not isinstance(dataset['obstype'], int):
        raise ValueError('obstype must be integer')

    if dataset['obstype'] not in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
        raise ValueError('unknown observable type (obstype = ' + str(dataset['obstype']) + ')')

    if dataset['cormat'] is not None:
        if not isinstance(dataset['cormat'], np.ndarray):
            raise ValueError('cormat must be an two-dimensional array (matrix)')
        if dataset['cormat'].shape[0] != dataset['cormat'].shape[1]:
            raise ValueError('cormat must be a square matrix')
        if dataset['cormat'].shape[0] < len(dataset['energies']):
            raise ValueError('number of rows in cormat must be larger than or equal number of datapoints')

    if len(dataset['reacids']) > 3:
        raise ValueError('number of reaction ids must be smaller or equal 3')

    if len(dataset['energies']) != len(dataset['measurements']):
        raise ValueError('number of energies must equal number of measurements')

    if dataset['partialuncs'].shape[0] != len(dataset['energies']):
        raise ValueError('number of rows in partialuncs must equal number of energies')

    if dataset['partialuncs'].shape[1] != 12:
        raise ValueError('partialuncs must contain exactly 12 columns')


    if 'NNCOX' not in dataset:
        raise ValueError('NNCOX must be zero')
    NNCOX = dataset['NNCOX']

    out = []

    # convenient abbreviations

    NS = dataset['dataset_id']
    MT = dataset['obstype']
    ECOR = dataset['cormat'] if 'cormat' in dataset else None 
    CO = dataset['partialuncs']
    # derived quantities
    NCOX = 0 if ECOR is None else ECOR.shape[0] 
    NCT = len(dataset['reacids']) 
    NT = dataset['reacids']
    IDEN = np.zeros(9)
    IDEN[3] = dataset['year']
    IDEN[4] = 1 if not 'dataset_tag' in dataset else dataset['dataset_tag']  # dataset_tag
    NCCS = len(dataset['correlated_datasets']) if 'correlated_datasets' in dataset else 0
    IDEN[5] = NCCS  # num of correlated datasets
    # NCSST: dataset numbers with cross-correlations to this dataset
    # NEC: error component pairs of cross correlations
    # FCFC: cross correlation factors
    NCSST = dataset['correlated_datasets'] if NCCS > 0 else None
    NEC = dataset['paired_unccomp_indices'] if 'paired_unccomp_indices' in dataset else None
    FCFC = dataset['cross_correlation_factors'] if 'cross_correlation_factors' in dataset else None 

    NENF = dataset['normunc_component_tags'] if 'normunc_component_tags' in dataset else np.zeros(10)
    ENFF = dataset['normunc_components'] if 'normunc_components' in dataset else np.zeros(10)

    EPAF = dataset['normunc_comp_params'] if 'normunc_comp_params' in dataset else np.zeros((3,11))
    NETG = dataset['endep_uncomp_tags'] if 'endep_uncomp_tags' in dataset else np.zeros(11)

    ref = dataset['reference'].rjust(32)
    BREF = [ref[:8], ref[8:16], ref[16:24], ref[24:32]]
    aut = dataset['author'].rjust(32)
    CLABL = [aut[:8], aut[8:16], aut[16:24], aut[24:32]]

    outNT = np.zeros(3)
    outNT[:len(NT)] = NT

    out.append(write_control_header('DATA', NS, MT, NCOX, NCT, outNT[0], outNT[1], outNT[2], NNCOX))

    format131 = "(3I5,4A8,4A8)"
    out.append(fort_write(None, format131, [[IDEN[3:6]], CLABL, BREF], retstr=True))
    out[-1] = out[-1].rstrip()

    if (MT == 2 or MT == 4 or MT == 8 or MT == 9):
        MTTP = 2
    else:
        MTTP = 1

    # write normalization components and associated tags
    if MTTP != 2:
        format201 = "(10F5.1,10I3)"
        out.append(fort_write(None, format201, [ENFF, NENF], retstr=True))

    # write energy dependent uncertainty parameters
    format202 = "(3F5.2,I3)"
    for K in range(11):
        out.append(fort_write(None, format202, [EPAF[:,K], NETG[K]], retstr=True))

    # write cross correlation info between datasets
    if NCCS != 0:
        for K in range(NCCS):  # .lbl204
            format205 = "(I5,20I3)"
            tmp = np.reshape(NEC[:,:, K], 20, order='F') 
            out.append(fort_write(None, format205, [NCSST[K], tmp], retstr=True))
            format841 = "(10F5.1)"
            out.append(fort_write(None, format841, FCFC[:,K], retstr=True))

    # write energies, cross sections and uncertainty components
    E = dataset['energies']
    CSS = dataset['measurements']
    CO = dataset['partialuncs']
    for KS in range(CO.shape[0]):
        format109 = format_dic.get('format109', "(2E10.4,12F5.1)")
        out.append(fort_write(None, format109, [E[KS], CSS[KS], CO[KS,:]], retstr=True))
    out.append(fort_write(None, format109, np.zeros(2 + len(CO[KS,:])), retstr=True))

    # write correlation matrix if given
    if NCOX != 0:
        format161 = "(10F8.5)"
        for KS in range(NCOX):  # .lbl61
            out.append(fort_write(None, format161, ECOR[KS,:(KS+1)], retstr=True))

    return '\n'.join(out)




def extract_dataset_from_datablock(ID, datablock):
    start_idx, end_idx = get_dataset_range(ID, datablock)
    NCOX = datablock.NCOX[ID] 
    author = ''.join(datablock.CLABL[ID,1:5])
    journal = ''.join(datablock.BREF[ID,1:5])
    dataset_id = datablock.IDEN[ID,6] 
    dataset_tag = datablock.IDEN[ID,4]
    year = datablock.IDEN[ID,3]
    obstype = datablock.IDEN[ID,7]
    cormat = datablock.userECOR[1:(NCOX+1),1:(NCOX+1)] if NCOX > 0 else None 
    NNCOX = datablock.NNCOX[ID]
    #if dataset_id == 8019:
    #    import pdb; pdb.set_trace()

    reacids = datablock.NT[ID,1:(datablock.NCT[ID]+1)]  
    energies = datablock.E[start_idx:(end_idx+1)]
    measurements = datablock.CSS[start_idx:(end_idx+1)]
    partialuncs = datablock.userCO[1:13, start_idx:(end_idx+1)].T
    correlated_datasets = datablock.NCSST[ID, 1:(datablock.IDEN[ID,5]+1)]
    paired_unccomp_indices = datablock.NEC[ID,1:3, 1:11,1:(len(correlated_datasets)+1)] 
    # if dataset_id == 8017:
    #     import pdb; pdb.set_trace()
    cross_correlation_factors = datablock.FCFC[ID,1:11,1:(len(correlated_datasets)+1)]
    normunc_components = datablock.ENFF[ID,1:11]
    normunc_comp_params = datablock.EPAF[1:4,1:12,ID]
    normunc_component_tags = datablock.NENF[ID,1:11]
    endep_uncomp_tags =  datablock.NETG[1:12, ID]
    return {
        'author': author,
        'reference': journal,
        'dataset_id': int(dataset_id),
        'dataset_tag': dataset_tag,
        'year': int(year),
        'obstype': int(obstype),
        'cormat': cormat,
        'reacids': reacids,
        'energies': energies,
        'measurements': measurements,
        'partialuncs': partialuncs,
        'correlated_datasets': correlated_datasets,
        'paired_unccomp_indices': paired_unccomp_indices,
        'cross_correlation_factors': cross_correlation_factors,
        'normunc_components': normunc_components,
        'normunc_comp_params': normunc_comp_params,
        'normunc_component_tags': normunc_component_tags,
        'endep_uncomp_tags': endep_uncomp_tags,
        'NNCOX': NNCOX
            }



def extract_experiment_datatable(datablock_list, priordf=None):
    res = None
    numpts1 = 0
    numpts2 = 0
    for datablock in datablock_list:
        for i in range(1,datablock.num_datasets+1):
            ds = extract_dataset_from_datablock(i, datablock)
            NT = len(ds['reacids'])
            curfrm = pd.DataFrame.from_dict({
                'author': ds['author'].strip(),
                'reference': ds['reference'].strip(),
                'dataset_id': ds['dataset_id'],
                'energy': ds['energies'],
                'measurement': ds['measurements'],
                'obstype': MT_to_label(ds['obstype']),
                #'xsid1': ds['reacids'][0],
                'reac1': priordf.loc[priordf['xsid']==ds['reacids'][0]].iloc[0]['reac'] if NT > 0 else '',
                #'xsid2': ds['reacids'][1] if len(ds['reacids']) >= 2 else 0,
                'reac2': priordf.loc[priordf['xsid']==ds['reacids'][1]].iloc[0]['reac'] if NT > 1 else '' ,
                #'xsid3': ds['reacids'][2] if len(ds['reacids']) >= 3 else 0
                'reac3': priordf.loc[priordf['xsid']==ds['reacids'][2]].iloc[0]['reac'] if NT > 2 else '',
                })
            numpts2 += curfrm.shape[0]
            if res is None:
                res = curfrm
            else:
                res = pd.concat([res, curfrm])

        numpts1 += datablock.num_datapoints
        if numpts1 != numpts2:
            raise IndexError('There is some strange mismatch of read elements')

    res.reset_index(inplace=True, drop=True)
    return res



def extract_prior_datatable(APR):
    NC = APR.NC
    resfrm = None 
    for xsid in range(1,NC+1):
        reac = ''.join(APR.CLAB[xsid, 1:3]).strip()
        start_index, end_index = get_prior_range(xsid, APR)
        curfrm = pd.DataFrame.from_dict({
            'xsid': xsid,
            'reac': reac, 
            'energy': APR.EN[start_index:(end_index+1)],
            'xs': APR.CS[start_index:(end_index+1)]
            })
        if resfrm is None:
            resfrm = curfrm
        else:
            resfrm = pd.concat([resfrm, curfrm])
    resfrm.reset_index(inplace=True, drop=True)
    return resfrm



def write_prior_text(prior_datatable):
    priordf = prior_datatable
    textarr = []

    num_reacs = priordf['xsid'].nunique()
    num_rows = len(priordf.index)
    newline = write_control_header('APRI', num_rows, num_reacs, 0, 0, 0, 0, 0, 0)  
    textarr.append(newline)
    for curxsid in range(1, num_reacs+1):
        curdf = priordf[priordf.xsid == curxsid]
        if curdf.reac.nunique() != 1:
            raise ValueError('xsid is associated with a single reaction')
        if (len(curdf.index) == 0):
            raise IndexError("The following xsid is missing: " + str(curxsid))
        curdf = curdf.sort_values(by='energy')
        curreac = curdf.iloc[0]['reac']
        # format120 = r"(2A8)"
        # APR.CLAB[K, 1:3] = fort_read(file_IO3, format120)
        textarr.append(curreac)
        format103 = "(2E10.4)"
        for curidx in range(len(curdf.index)):
            cur_en = curdf.iloc[curidx]['energy']
            cur_xs = curdf.iloc[curidx]['xs']
            newline = fort_write(None, format103, [cur_en, cur_xs], retstr=True)
            textarr.append(newline)
        newline = fort_write(None, format103, [0, 0], retstr=True)
        textarr.append(newline)
    text = '\n'.join(textarr)
    return text



def extract_fission_datatable(fisdata):
    num = fisdata.NFIS
    resdt = pd.DataFrame.from_dict({
        'energy': fisdata.ENFIS[1:(num+1)],
        'xs': fisdata.FIS[1:(num+1)]
        })
    return resdt



def write_fission_text(fission_datatable):
    dt = fission_datatable
    format119 = "(2E13.5)"
    res = []
    newline = write_control_header('FIS*', 1, 0, 0, 0, 0, 0, 0, 0)
    res.append(newline)
    for idx in range(len(dt.index)):
        cur_en = dt['energy'][idx]
        cur_xs = dt['xs'][idx]
        newline = fort_write(None, format119, [cur_en, cur_xs], retstr=True)
        res.append(newline)
    newline = fort_write(None, format119, [0, 0], retstr=True)
    res.append(newline)
    text = '\n'.join(res)
    return text

