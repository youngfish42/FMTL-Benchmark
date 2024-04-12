from torch.utils.data import DataLoader

from .utils.custom_collate import collate_mil


def get_dataset(dataname, train, transform, tasks, dataidxs=None):
    if dataname == 'pascalcontext':
        from .pascal_context import PASCALContext
        database = PASCALContext(train=train, transform=transform, tasks=tasks, dataidxs=dataidxs)
    elif dataname == 'nyud':
        from .nyud import NYUD
        database = NYUD(train=train, transform=transform, tasks=tasks, dataidxs=dataidxs)
    elif dataname == 'cityscapes':
        from .cityscapes import CITYSCAPES
        database = CITYSCAPES(train=train, task_list=tasks, dataidxs=dataidxs)
    else:
        raise NotImplementedError("dataname: Choose among pascalcontext and nyud")

    return database


def get_dataloader(train, configs, dataset, sampler=None):
    if train:
        dataloader = DataLoader(dataset,
                                batch_size=configs['tr_batch'],
                                drop_last=True,
                                num_workers=configs['nworkers'],
                                collate_fn=collate_mil,
                                pin_memory=True,
                                sampler=sampler)
    else:
        dataloader = DataLoader(dataset,
                                batch_size=configs['val_batch'],
                                shuffle=False,
                                drop_last=False,
                                num_workers=configs['nworkers'],
                                collate_fn=collate_mil,
                                pin_memory=True)
    return dataloader
