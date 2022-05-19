import numpy as np

class Dataset(object):

    def __init__(self, dataset_dic=None):
        if dataset_dic is None:
            dataset_dic = {}
        self.dataset_dic = dataset_dic
        self._update_dic_skeleton()

    def _update_dic_skeleton(self):
        if 'type' not in self.dataset_dic:
            self.dataset_dic['type'] = 'legacy-experiment-dataset'
        # initialize the CO array
        if 'CSS' in self.dataset_dic and 'CO' not in self.dataset_dic:
            numpoints = len(self.dataset_dic['CSS'])
            CO = [[0. for i in range(12)] for j in range(numpoints)]
            self.dataset_dic['CO'] = CO
        # initialize the ENFF array
        if 'ENFF' not in self.dataset_dic: 
            ENFF = [0. for i in range(10)] 
            self.dataset_dic['ENFF'] = ENFF
        # initialize the tags of the normalization components
        if 'NENF' not in self.dataset_dic:
            NENF = [0 for i in range(10)] 
            self.dataset_dic['NENF'] = NENF
        # initialize the NNCOX number
        if 'NNCOX' not in self.dataset_dic:
            self.dataset_dic['NNCOX'] = 0
        # initialize NETG with all zeros
        if 'NETG' not in self.dataset_dic:
            NETG = [0 for i in range(11)]
            self.dataset_dic['NETG'] = NETG
        # initialize the EPAF array
        if 'EPAF' not in self.dataset_dic:
            EPAF = [[0.0 for i in range(11)] for j in range(3)] 
            self.dataset_dic['EPAF'] = EPAF

    def _is_float_list(self, l):
        if not isinstance(l, list):
            return False
        return all([isinstance(v, float) for v in l])

    def _is_int_list(self, l):
        if not isinstance(l, list):
            return False
        return all([isinstance(v, int) for v in l])

    def define_metadata(self, dataset_id, year=0,
            author='anonymous', reference='unknown'):
        if not isinstance(dataset_id, int):
            raise TypeError('dataset_id must be integer')
        if not isinstance(year, int):
            raise TypeError('year must be integer')
        if not isinstance(author, str):
            raise TypeError('author must be a string')
        if not isinstance(reference, str):
            raise TypeError('reference must be a string')

        self.dataset_dic.update({
            'type': 'legacy-experiment-dataset',
            'NS': dataset_id,
            'YEAR': year,
            'TAG': 1, 
            'CLABL': author,
            'BREF': reference
            })

    def define_quantity(self, mt, reac_ids):
        if not isinstance(mt, int):
            raise TypeError('mt must be an integer')
        if not self._is_int_list(reac_ids):
            raise TypeError('reac_ids must be list of integers')
        self.dataset_dic.update({
            'MT': mt,
            'NT': reac_ids
            })

    def define_measurements(self, energies, datapoints):
        if not self._is_float_list(energies):
            raise TypeError('all energies must be float')
        if not self._is_float_list(datapoints):
            raise TypeError('all datapoints must be float')
        self.dataset_dic.update({
            'E': energies,
            'CSS': datapoints
            })
        self._update_dic_skeleton()

    def add_norm_uncertainty(self, uncertainty):
        if not isinstance(uncertainty, float):
            raise TypeError('uncertainty must be a float')
        ENFF = self.dataset_dic['ENFF']
        NENF = self.dataset_dic['NENF']
        col_idx = 0
        while col_idx < len(ENFF) and ENFF[col_idx] != 0:
            col_idx += 1
        if col_idx == len(ENFF):
            raise IndexError('no uncertainty slot available anymore')
        ENFF[col_idx] = uncertainty

    def add_uncertainties(self, uncertainties, epaf_params=(0.5,0.5,0.5)):
        if not self._is_float_list(uncertainties):
            raise TypeError('all uncertainties must be float')
        if 'CSS' not in self.dataset_dic:
            raise IndexError('please define first the measurements ' +
                             'via the method define_measurements')
        numpoints = len(self.dataset_dic['CSS'])
        if len(uncertainties) != numpoints:
            raise IndexError(f'number of uncertainties ({len(uncertainties)}) '
                             f'does not match number of datapoints ({numpoints})')
        # find first available uncertainty column
        CO = self.dataset_dic['CO']
        NETG = self.dataset_dic['NETG']
        EPAF = self.dataset_dic['EPAF']
        col_idx = 2
        while col_idx <= 10:
            if all([CO[i][col_idx]==0 for i in range(numpoints)]):
                break
            col_idx += 1
        if col_idx > 10:
            raise IndexError('no uncertainy slot available anymore') 
        # add the uncertainties
        NETG[col_idx] = 1
        EPAF[0][col_idx] = epaf_params[0]
        EPAF[1][col_idx] = epaf_params[1]
        EPAF[2][col_idx] = epaf_params[2]
        for i, unc in enumerate(uncertainties):
            CO[i][col_idx] = unc
        # save the modified uncertainty array   
        self.dataset_dic['CO'] = CO
        self.dataset_dic['NETG'] = NETG 
        self.dataset_dic['EPAF'] = EPAF

    def get_author(self):
        return self.dataset_dic['CLABL'].strip()

    def get_mtnum(self):
        return self.dataset_dic['MT']

    def get_dataset_id(self):
        return self.dataset_dic['NS']

    def get_numpoints(self):
        return len(self.dataset_dic['CSS'])

    def get_cross_sections(self):
        return tuple(self.dataset_dic['CSS'])

    def get_energies(self):
        return tuple(self.dataset_dic['E'])

