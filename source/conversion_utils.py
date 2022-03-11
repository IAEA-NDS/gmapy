import json
from collections import OrderedDict
import numpy as np
from gmap_snippets import get_dataset_range
from data_management import init_datablock, init_fisdata



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



def sanitize_datablock(datablock):
    """Creates a clean and beautiful datablock object."""
    data = datablock
    num_datasets = data.num_datasets
    num_datapoints = data.num_datapoints

    dataset_list = []

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
        CLABL = data.CLABL[dsidx, 1:5]
        BREF = data.BREF[dsidx, 1:5]

        YEAR = data.IDEN[dsidx, 3]
        TAG = data.IDEN[dsidx, 4]

        # beautification of CLABL and BREF
        # CLABL = ''.join(CLABL).strip()
        # BREF = ''.join(BREF).strip()

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
            # correlation matrix 
            ECOR = data.userECOR[1:(NCOX+1), 1:(NCOX+1)]

        #  total uncertainty
        DCS = data.DCS[1:(curnumpts+1)]

        # construct the output object
        dataset = OrderedDict()
        computed = OrderedDict()
        dataset['NS'] = NS
        dataset['MT'] = MT
        dataset['YEAR'] = YEAR
        dataset['TAG'] = TAG
        computed['NCT'] = NCT
        dataset['NT'] = NT
        dataset['NCOX'] = NCOX
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
            dataset['ECOR'] = ECOR

        computed['DCS'] = DCS

        dataset['computed'] = computed

        dataset_list.append(dataset)

    return dataset_list



def desanitize_datablock(datablock):
    """Convert sanitized datablock to raw one."""
    data = init_datablock()
    data.num_datasets = len(datablock)
    data.num_datapoints = 0
    start_idx = 1
    # NOTE: Fortran GMAP allows several choices of MODC
    # but as the standards database only relies on MODC=3
    # it is hardcoded here
    data.MODC = 3

    for tid, dataset in enumerate(datablock):
        ID = tid+1
        ds = dataset
        numpts = len(ds['CSS'])
        if len(ds['E']) != len(ds['CSS']):
            raise ValueError('energy mesh in E and number of cross sections CSS must match')

        data.num_datapoints += numpts

        data.NCT[ID] = len(ds['NT'])
        data.NT[ID,1:(data.NCT[ID]+1)] = ds['NT']
        NCOX = ds['NCOX']
        data.NCOX[ID] = NCOX
        data.NNCOX[ID] = ds['NNCOX']
        data.IDEN[ID,2] = start_idx
        data.IDEN[ID,3] = ds['YEAR']
        data.IDEN[ID,4] = ds['TAG']

        data.IDEN[ID,6] = ds['NS']
        data.IDEN[ID,7] = ds['MT']

        data.MTTP[ID] = 2 if ds['MT'] in (2,4,8,9) else 1
        data.IDEN[ID,8] = data.MTTP[ID]
        data.CLABL[ID,1:5] = ds['CLABL']
        data.BREF[ID,1:5] = ds['BREF']

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
        data.userCO[1:13, start_idx:fidx] = ds['CO'].T
        # data.userCO is CO as provided by user
        # and data.CO may be changed due to 
        # Axton special below
        data.CO[1:13, start_idx:fidx] = data.userCO[1:13, start_idx:fidx]

        data.IDEN[ID,1] = numpts

        if NCOX != 0:
            data.userECOR[1:(NCOX+1),1:(NCOX+1)] = ds['ECOR']

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
            if isinstance(val1, np.ndarray):
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

