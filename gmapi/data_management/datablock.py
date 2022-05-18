from .dataset import Dataset


class Datablock(object):

    def __init__(self):
        self.datablock_dic = {}
        self.datablock_dic['type'] = 'legacy-experiment-datablock'
        self.datablock_dic['datasets'] = []
        self.dataset_list = []

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
        for cur_idx, curdataset in enumerate(self.dataset_list):
            cur_id = curdataset.get_dataset_id()
            if cur_id in dataset_ids:
                remove_idcs.append(cur_idx)
        if len(remove_idcs) != len(dataset_ids) :
            raise IndexError('not all dataset_ids were found in the datablock')
        remove_idcs.reverse()
        for cur_idx in remove_idcs:
            del self.dataset_list[cur_idx]
            del self.datablock_dic['datasets'][cur_idx]

    def list_dataset_ids(self):
        dslist = self.dataset_list
        return [d.get_dataset_id() for d in dslist]

    def get_datablock_dic(self):
        return self.datablock_dic

