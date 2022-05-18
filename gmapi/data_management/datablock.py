from .dataset import Dataset


class Datablock(object):

    def __init__(self):
        self.dataset_list = []

    def add_datasets(self, datasets):
        if isinstance(datasets, Dataset):
            datasets = [datasets] 
        if not all([isinstance(d, Dataset) for d in datasets]):
            raise TypeError('all items in the list must be Datasets')
        self.dataset_list.extend(datasets)

    def remove_datasets(self, dataset_ids):
        if isinstance(dataset_ids, int):
            dataset_ids = [dataset_ids]
        if not all([isinstance(v, int) for v in dataset_ids]):
            raise TypeError('all items in the list must be integers')
        remove_idcs = []
        for cur_idx, curdataset in enumerate(self.dataset_list):
            cur_id = curdataset.get_dataset_id()
            if cur_id in dataset_ids:
                remove_idcs.append(cur_idx)
        if len(remove_idcs) != len(dataset_ids) :
            raise IndexError('not all dataset_ids were found in the datablock')
        remove_idcs.reverse()
        for cur_idx in remove_idcs:
            del self.dataset_list[cur_idx]

    def list_dataset_ids(self):
        dslist = self.dataset_list
        return [d.get_dataset_id() for d in dslist]

