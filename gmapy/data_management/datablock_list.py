from .datablock import Datablock


class DatablockList(object):

    def __init__(self, datablock_list=[]):
        dblist = []
        for db in datablock_list:
            if isinstance(db, dict):
                db = Datablock(db) 
            dblist.append(db)
        self.datablock_list = dblist

    def __getitem__(self, index):
        return self.datablock_list[index]

    def add_datablock(self, datablock):
        if not isinstance(datablock, Datablock):
            raise TypeError('argument must be a Datablock')
        self.datablock_list.append(datablock)

    def remove_datasets(self, dataset_ids): 
        if isinstance(dataset_ids, int):
            dataset_ids = [dataset_ids]
        if not all([isinstance(v, int) for v in dataset_ids]):
            raise TypeError('all items in the list must be integers')
        db_rem_idcs = []
        for idx, db in enumerate(self.datablock_list):
            cur_ids = db.list_dataset_ids()
            remove_ids = [dsid for dsid in dataset_ids if dsid in cur_ids]
            if len(remove_ids) > 0:
                db.remove_datasets(remove_ids)
            # if the dataset is now empty
            # we remove the complete dataset
            if db.get_numpoints() == 0:
                db_rem_idcs.append(idx)
        db_rem_idcs.reverse()
        for idx in db_rem_idcs:
            del self.datablock_list[idx]

    def remove_datasets_by_mtnums(self, mtnums): 
        if isinstance(mtnums, int):
            mtnums = [mtnums]
        if not all([isinstance(v, int) for v in mtnums]):
            raise TypeError('all items in the list must be integers')
        db_rem_idcs = []
        for idx, db in enumerate(self.datablock_list):
            db.remove_datasets_by_mtnums(mtnums)
            # if the dataset is now empty
            # we remove the complete dataset
            if db.get_numpoints() == 0:
                db_rem_idcs.append(idx)
        db_rem_idcs.reverse()
        for idx in db_rem_idcs:
            del self.datablock_list[idx]

    def list_dataset_ids(self):
        res = []
        for db in self.datablock_list:
            res.extend(db.list_dataset_ids())
        return tuple(res)

    def get_datablock_list(self, ret_type=dict):
        if ret_type == dict:
            return [db.get_datablock_dic() for db in self.datablock_list]
        elif ret_type == Datablock:
            return self.datablock_list
        else:
            raise ValueError('ret_type must be dict or Datablock')

