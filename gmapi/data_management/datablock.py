from .dataset import Dataset
from .uncfuns import create_relunc_vector, create_experimental_covmat
from scipy.sparse import csr_matrix
import numpy as np
import warnings


class Datablock(object):

    def __init__(self, datablock_dic=None):
        if datablock_dic is None:
            self.datablock_dic = {}
            self.datablock_dic['type'] = 'legacy-experiment-datablock'
            self.datablock_dic['datasets'] = []
            self.dataset_list = []
        else:
            if datablock_dic['type'] != 'legacy-experiment-datablock':
                raise TypeError('invalid datablock dictionary: ' +
                                'type must be "legacy-experiment-datablock')
            self.datablock_dic = datablock_dic
            self.dataset_list = [Dataset(ds) for ds in datablock_dic['datasets']]

    def add_datasets(self, datasets):
        if isinstance(datasets, Dataset):
            datasets = [datasets]
        if not all([isinstance(d, Dataset) for d in datasets]):
            raise TypeError('all items in the list must be Datasets')
        dslist = self.dataset_list
        dslist.extend(datasets)
        dslist2 = self.datablock_dic['datasets']
        dslist2.extend([d.dataset_dic for d in datasets])

    def remove_datasets(self, dataset_ids):
        if isinstance(dataset_ids, int):
            dataset_ids = [dataset_ids]
        if not all([isinstance(v, int) for v in dataset_ids]):
            raise TypeError('all items in the list must be integers')
        remove_idcs = []
        dslist = self.datablock_dic['datasets']
        # variables to keep track of the
        # indices of the datapoints covered by the datasets
        point_idcs = []
        cur_point_idx = 0
        for cur_idx, curdataset in enumerate(self.dataset_list):
            cur_id = curdataset.get_dataset_id()
            curnumpts = curdataset.get_numpoints()
            next_point_idx = cur_point_idx + curnumpts
            if cur_id in dataset_ids:
                remove_idcs.append(cur_idx)
                point_idcs.append((cur_point_idx, next_point_idx))
            cur_point_idx = next_point_idx

        if len(remove_idcs) != len(dataset_ids):
            raise IndexError('not all dataset_ids were found in the datablock')
        remove_idcs.reverse()
        for cur_idx in remove_idcs:
            del self.dataset_list[cur_idx]
            del self.datablock_dic['datasets'][cur_idx]

        # delete corresponding rows and columns in
        # datablock correlation matrix if present
        ptmask = np.full(cur_point_idx, True)
        for start_idx, stop_idx in point_idcs:
            ptmask[start_idx:stop_idx] = False
        if 'ECOR' in self.datablock_dic:
            new_ecor = np.array(self.datablock_dic['ECOR'])
            new_ecor = new_ecor[ptmask,:]
            new_ecor = new_ecor[:,ptmask]
            self.datablock_dic['ECOR'] = list(new_ecor)

    def remove_datasets_by_mtnums(self, mtnums):
        if isinstance(mtnums, int):
            mtnums = [mtnums]
        if not all([isinstance(v, int) for v in mtnums]):
            raise TypeError('all items in the list must be integers')
        remove_idcs = []
        dslist = self.datablock_dic['datasets']
        for cur_idx, curdataset in enumerate(self.dataset_list):
            cur_mtnum = curdataset.get_mtnum()
            if cur_mtnum in mtnums:
                remove_idcs.append(cur_idx)
        remove_idcs.reverse()
        for cur_idx in remove_idcs:
            del self.dataset_list[cur_idx]
            del self.datablock_dic['datasets'][cur_idx]

    def define_correlation_matrix(self, cormat):
        if len(cormat.shape) != 2 or cormat.shape[0] != cormat.shape[1]:
            raise TypeError('cormat must be a square atrix')
        size = self.get_numpoints()
        m = cormat.shape[0]
        if m != size:
            raise IndexError(f'Dimension of matrix ({m} x {m}) must '
                             f'match number of datapoints in the datablock '
                             f'({size})')
        warnings.warn('Correlation matrix will override any correlations ' +
                      'defined at the level of datasets')
        self.datablock_dic['ECOR'] = cormat

    def list_dataset_ids(self):
        dslist = self.dataset_list
        return [d.get_dataset_id() for d in dslist]

    def get_numpoints(self):
        sum_numpoints = 0
        for ds in self.dataset_list:
            sum_numpoints += ds.get_numpoints()
        return sum_numpoints

    def get_cross_sections(self):
        res = []
        for ds in self.dataset_list:
            res.extend(ds.get_cross_sections())
        return tuple(res)

    def get_energies(self):
        res = []
        for ds in self.dataset_list:
            res.extend(ds.get_energies())
        return tuple(res)

    def get_uncertainties(self, unit='percent'):
        if unit != 'percent':
            raise ValueError('unit must be percent')
        return create_relunc_vector([self.datablock_dic])

    def get_covariance_matrix(self, unit='percent'):
        css = np.array(self.get_cross_sections())
        covmat = create_experimental_covmat(
                [self.datablock_dic], propcss=css)
        if unit == 'absolute':
            return csr_matrix(covmat)
        elif unit == 'relative' or 'percent':
            tmp = 1/css
            if 'precent':
                tmp *= 100
            res = (covmat.toarray() * tmp.reshape(-1, 1)) * tmp.reshape(1,-1)
            return csr_matrix(res)
        else:
            raise ValueError('unit must be absolute, relative or percent')

    def get_correlation_matrix(self):
        covmat = self.get_covariance_matrix()
        uncs = np.sqrt(covmat.diagonal())
        covmat /= uncs.reshape(-1, 1)
        covmat /= uncs.reshape(1, -1)
        cormat = covmat
        return cormat

    def get_datablock_dic(self):
        return self.datablock_dic

