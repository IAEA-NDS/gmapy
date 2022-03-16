import json
import os
from collections import OrderedDict
import numpy as np
from gmap_snippets import get_dataset_range, get_prior_range
from data_management import init_datablock, init_fisdata, init_prior
from generic_utils import Bunch
from database_reading import read_gma_database
from mappings.helperfuns import SHAPE_MT_IDS



class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)



def convert_GMA_database_to_JSON(dbfile, jsonfile):
    if not os.path.exists(dbfile):
        raise FileNotFoundError('The GMA database file %s does not exist'%dbfile)
    if os.path.exists(jsonfile):
        raise OSError('The file %s already exists. Aborting'%jsonfile)
    db_dic = read_gma_database(dbfile)
    APR = db_dic['APR']
    datablock_list = db_dic['datablock_list']
    jsondic = OrderedDict()
    jsondic['prior'] = sanitize_prior(APR)
    jsondic['datablocks'] = [sanitize_datablock(b) for b in datablock_list]
    jsonstr = json.dumps(jsondic, cls=NumpyEncoder, indent=4, allow_nan=False)
    with open(jsonfile, 'w') as f:
        f.write(jsonstr)
    return



def sanitize_datablock(datablock):
    """Creates a clean and beautiful datablock object."""
    data = datablock
    num_datasets = data.num_datasets
    num_datapoints = data.num_datapoints

    dataset_list = []
    new_datablock = OrderedDict()
    new_datablock['type'] = 'legacy-experiment-datablock'
    new_datablock['datasets'] = dataset_list

    for dsidx in range(1, num_datasets+1):
        sidx, fidx = get_dataset_range(dsidx, data)
        curnumpts = fidx - sidx + 1

        NS = data.IDEN[dsidx, 6]
        MT = data.IDEN[dsidx, 7]
        NCT = data.NCT[dsidx]
        NT = data.NT[dsidx, 1:(NCT+1)]
        NNCOX = data.NNCOX[dsidx]
        NCOX = data.NCOX[dsidx]
        ofs = data.IDEN[dsidx, 2]
        CLABL = ''.join(data.CLABL[dsidx, 1:5])
        BREF = ''.join(data.BREF[dsidx, 1:5])

        YEAR = data.IDEN[dsidx, 3]
        TAG = data.IDEN[dsidx, 4]

        # energy dependent uncertainty parameters
        EPAF = data.EPAF[1:4, 1:12, dsidx]
        NETG = data.NETG[1:12, dsidx]

        # MTTP ... flag for shape data (1: absolute, 2: shape)
        MTTP = data.MTTP[dsidx]

        if MTTP != 2:  # for non-shape data
            # normalization uncertainties
            ENFF = data.ENFF[dsidx, 1:11]
            NENF = data.NENF[dsidx, 1:11]

        NCCS = data.IDEN[dsidx, 5]
        if NCCS != 0:
            # correlation information
            NCSST = data.NCSST[dsidx, 1:(NCCS+1)]
            NEC = data.NEC[dsidx, 1:3, 1:11, 1:(NCCS+1)]
            FCFC = data.FCFC[dsidx, 1:11, 1:(NCCS+1)]


        # energy mesh, cross sections, and uncertainty components
        E = data.E[sidx:(fidx+1)]
        CSS = data.CSS[sidx:(fidx+1)]
        CO = data.userCO[1:13, sidx:(fidx+1)]

        if NCOX != 0:
            if dsidx != num_datasets:
                raise ValueError('NCOX must be zero except for the last dataset')
            if NCOX != num_datapoints:
                raise IndexError('dimension of correlation matrix does not ' +
                        'match number of datapoints')
            # correlation matrix 
            ECOR = data.userECOR[1:(NCOX+1), 1:(NCOX+1)]

        #  total uncertainty
        DCS = data.DCS[1:(curnumpts+1)]

        # construct the output object
        dataset = OrderedDict()
        computed = OrderedDict()
        dataset['type'] = 'legacy-experiment-dataset'
        dataset['NS'] = NS
        dataset['MT'] = MT
        dataset['YEAR'] = YEAR
        dataset['TAG'] = TAG
        computed['NCT'] = NCT
        dataset['NT'] = NT
        dataset['NNCOX'] = NNCOX
        computed['ofs'] = ofs
        dataset['CLABL'] = CLABL
        dataset['BREF'] = BREF

        dataset['EPAF'] = EPAF
        dataset['NETG'] = NETG

        computed['MTTP'] = MTTP
        if MTTP != 2:
            dataset['ENFF'] = ENFF
            dataset['NENF'] = NENF

        if NCCS != 0:
            dataset['NCSST'] = NCSST
            dataset['NEC'] = NEC
            dataset['FCFC'] = FCFC
        computed['NCCS'] = NCCS

        dataset['E'] = E
        dataset['CSS'] = CSS
        dataset['CO'] = CO.T

        if NCOX != 0:
            new_datablock['ECOR'] = ECOR

        computed['DCS'] = DCS

        dataset['computed'] = computed

        dataset_list.append(dataset)

    return new_datablock



def desanitize_datablock(datablock):
    """Convert sanitized datablock to raw one."""
    if datablock['type'] != 'legacy-experiment-datablock':
        raise TypeError('Type must be legacy-experiment-datablock')

    data = init_datablock()
    data.num_datapoints = 0
    start_idx = 1
    # NOTE: Fortran GMAP allows several choices of MODC
    # but as the standards database only relies on MODC=3
    # it is hardcoded here
    data.MODC = 3

    gNCOX = 0
    if 'ECOR' in datablock:
        Ecor = np.array(datablock['ECOR'])
        if Ecor.shape[0] != Ecor.shape[1]:
            raise IndexError('Ecor is not square matrix')
        gNCOX = Ecor.shape[0]

    dataset_list = datablock['datasets']
    data.num_datasets = len(dataset_list)
    for tid, dataset in enumerate(dataset_list):
        ID = tid+1
        ds = dataset
        numpts = len(ds['CSS'])
        if len(ds['E']) != len(ds['CSS']):
            raise ValueError('energy mesh in E and number of cross sections CSS must match')

        data.num_datapoints += numpts

        data.NCT[ID] = len(ds['NT'])
        data.NT[ID,1:(data.NCT[ID]+1)] = ds['NT']
        NCOX = gNCOX if tid == data.num_datasets-1 else 0
        data.NCOX[ID] = NCOX
        data.NNCOX[ID] = ds['NNCOX']
        data.IDEN[ID,2] = start_idx
        data.IDEN[ID,3] = ds['YEAR']
        data.IDEN[ID,4] = ds['TAG']

        data.IDEN[ID,6] = ds['NS']
        data.IDEN[ID,7] = ds['MT']

        data.MTTP[ID] = 2 if ds['MT'] in SHAPE_MT_IDS else 1
        data.IDEN[ID,8] = data.MTTP[ID]
        z = ds['CLABL']
        data.CLABL[ID,1:5] = [z[:8], z[8:16], z[16:24], z[24:32]]
        z = ds['BREF']
        data.BREF[ID,1:5] = [z[:8], z[8:16], z[16:24], z[24:32]]

        NCCS = 0
        if 'NCSST' in ds:
            NCCS = len(ds['NCSST'])

        data.IDEN[ID,5] = NCCS

        if data.MTTP[ID] != 2:
            data.ENFF[ID, 1:11] = ds['ENFF']
            data.NENF[ID, 1:11] = ds['NENF']

        data.EPAF[1:4, 1:12, ID] = ds['EPAF']
        data.NETG[1:12, ID] = ds['NETG']

        if NCCS != 0:
            data.NCSST[ID, 1:(NCCS+1)] = ds['NCSST']
            data.NEC[ID, 1:3, 1:11, 1:(NCCS+1)] = ds['NEC']
            data.FCFC[ID, 1:11, 1:(NCCS+1)] = ds['FCFC']

        fidx = start_idx + numpts
        data.E[start_idx:fidx] = ds['E']
        data.CSS[start_idx:fidx] = ds['CSS']
        data.userCO[1:13, start_idx:fidx] = np.array(ds['CO']).T
        # data.userCO is CO as provided by user
        # and data.CO may be changed due to 
        # Axton special below
        data.CO[1:13, start_idx:fidx] = data.userCO[1:13, start_idx:fidx]

        data.IDEN[ID,1] = numpts

        if NCOX != 0:
            data.userECOR[1:(NCOX+1),1:(NCOX+1)] = Ecor

        XNORU = 0.
        if data.MTTP[ID] != 2:
            # calculate total normalization uncertainty
            XNORU = np.sum(np.square(ds['ENFF']))

        for NADD in range(start_idx, data.num_datapoints+1):
            # Axton special: downweight if NNCOX flag set
            if data.NNCOX[ID] != 0:
                for LZ in range(1,12):
                    data.CO[LZ, NADD] = data.userCO[LZ, NADD] / 10


            # calculate total uncertainty
            RELU = np.sum(np.square(data.CO[3:12, NADD]))

            data.DCS[NADD] = np.sqrt(XNORU + RELU)
            data.effDCS[NADD] = data.DCS[NADD]

        start_idx += numpts

    return data



def sanitize_fission_spectrum_block(fisblock):
    """Create a beautiful fission spectrum block from legacy one."""
    NFIS = fisblock.NFIS
    new_fisblock = OrderedDict()
    new_fisblock['type'] = 'legacy-fission-spectrum'
    new_fisblock['ENFIS'] = fisblock.ENFIS[1:(NFIS+1)] 
    new_fisblock['FIS'] = fisblock.FIS[1:(NFIS+1)]
    return new_fisblock



def desanitize_fission_spectrum_block(fisblock):
    """Create legacy fission spectrum data from new block structure."""
    if len(fisblock['ENFIS']) != len(fisblock['FIS']):
        raise IndexError('FIS and ENFIS in fisblock must be of same length')
    fisdata = init_fisdata()
    NFIS = len(fisblock['ENFIS'])
    fisdata.NFIS = NFIS
    fisdata.ENFIS[1:(NFIS+1)] = fisblock['ENFIS']
    fisdata.FIS[1:(NFIS+1)] = fisblock['FIS']
    return fisdata



def sanitize_prior(APR):
    """Convert legacy APR structure to new one."""
    blocklist = []
    for curid in range(1, APR.NC+1):
        sidx, fidx = get_prior_range(curid, APR)
        curblock = OrderedDict()
        curblock['type'] = 'legacy-prior-cross-section'
        curblock['ID'] = curid
        curblock['CLAB'] = APR.CLAB[curid, 1]
        curblock['EN'] = APR.EN[sidx:(fidx+1)]
        curblock['CS'] = APR.CS[sidx:(fidx+1)]
        blocklist.append(curblock)
    if APR.fisdata is not None:
        fisblock = sanitize_fission_spectrum_block(APR.fisdata)
        blocklist.append(fisblock)
    return blocklist



def desanitize_prior(priorlist):
    # create ID-datablock mapping
    id_dic = {}
    has_fisdata = False
    num_priorblocks = 0
    for curblock in priorlist:
        if curblock['type'] == 'legacy-fission-spectrum':
            if has_fisdata:
                raise IndexError('Only one legacy fission spectrum allowed')
            else:
                has_fisdata = True
                id_dic['fis'] = curblock
        elif curblock['type'] == 'legacy-prior-cross-section':
            ID = curblock['ID']
            if ID in id_dic:
                raise IndexError('ID %d exists multiple times in prior'%(curblock['ID'],))
            else:
                id_dic[ID] = curblock
                num_priorblocks += 1
    # read the prior cross sections
    APR = init_prior()
    totnumpts = 0
    cur_start_idx = 1
    for curid in range(1, num_priorblocks+1):
        if curid not in id_dic:
            raise IndexError('ID %d does not exist in prior but should')
        curblock = id_dic[curid]
        if len(curblock['EN']) != len(curblock['CS']):
            raise IndexError('In prior block with ID %d, ' +
                             'the lengths of EN and CS do not match')
        curnumpts = len(curblock['EN'])
        totnumpts += curnumpts
        cur_end_idx = cur_start_idx + curnumpts - 1
        APR.CLAB[curid, 1] = curblock['CLAB']
        APR.MCS[curid, 1] = curnumpts
        APR.MCS[curid, 2] = cur_start_idx
        APR.MCS[curid, 3] = cur_end_idx
        APR.EN[cur_start_idx:(cur_end_idx+1)] = curblock['EN']
        APR.CS[cur_start_idx:(cur_end_idx+1)] = curblock['CS']
        cur_start_idx += curnumpts

    APR.NR = totnumpts
    APR.NC = num_priorblocks
    if has_fisdata:
        APR.fisdata = desanitize_fission_spectrum_block(id_dic['fis'])
    return APR



def augment_datablocks_with_NTOT(datablock_list):
    NTOT = 0
    for datablock in datablock_list:
        NTOT += datablock.num_datapoints
        datablock.NTOT = NTOT



def compare_legacy_datablock_lists(list1, list2):
    """Compare two datablock lists for equivalence."""
    for dbidx in range(len(list1)):
        dbblock1 = list1[dbidx]
        dbblock2 = list2[dbidx]

        for key in dbblock1.__dict__:
            val1 = dbblock1.__dict__[key]
            val2 = dbblock2.__dict__[key]

            is_equal = False
            if type(val1) == Bunch:
                if not type(val1) == type(val2):
                    raise ValueError('Value mismatch for ' + str(key) +
                            '(%s vs %s)'%(type(val1),type(val2)))
                is_equal = compare_legacy_datablock_lists([val1], [val2])

            elif isinstance(val1, np.ndarray):
                if len(val1) != len(val2):
                    raise IndexError('Length mismatch')
                elif val1.dtype.type in (np.string_, np.object_):
                    is_equal = np.all(val1 == val2)
                else:
                    if key in ('CO', 'userCO'):
                        redvals1 = val1[:,1:(dbblock1.num_datapoints+1)]
                        redvals2 = val2[:,1:(dbblock2.num_datapoints+1)]
                        is_equal = np.all(np.isclose(redvals1, redvals2,
                            atol=0, rtol=1e-14))
                    else:
                        is_equal = np.all(np.isclose(val1, val2,
                            atol=0, rtol=1e-14))
            else:
                is_equal = val1 == val2

            if not is_equal:
                print(key + ' differs')
                import pdb; pdb.set_trace()
                break

    return is_equal

