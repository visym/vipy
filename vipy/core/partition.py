import random
    

def kfold(dataset, folds=2, randomize=False):
    raise NotImplementedError('FIXME')    

def leave_one_out(dataset, randomize=False, stratify=False):
    return [((set(dataset).difference(set(dataset[k]))), dataset[k]) for k in range(0,len(dataset))]    

