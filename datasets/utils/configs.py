PASCAL_OUT_CHANNELS = {'semseg': 21, 'human_parts': 7, 'normals': 3, 'edge': 1, 'sal': 2}

NYUD_OUT_CHANNELS = {'semseg': 40, 'normals': 3, 'edge': 1, 'depth': 1}

CITYSCAPES_OUT_CHANNELS = {'semseg': 19, 'depth': 1}

TRAIN_SCALE = {'pascalcontext': (512, 512), 'nyud': (448, 576), 'cityscapes': (768, 1536)}

TEST_SCALE = {'pascalcontext': (512, 512), 'nyud': (448, 576), 'cityscapes': (768, 1536)}

NUM_TRAIN_IMAGES = {'pascalcontext': 4998, 'nyud': 795, 'cityscapes': 2975}

NUM_TEST_IMAGES = {'pascalcontext': 5105, 'nyud': 654, 'cityscapes': 500}

NYUD_TASKS = ['semseg', 'normals', 'edge', 'depth']
PASCAL_TASKS = ['semseg', 'human_parts', 'normals', 'edge', 'sal']
ALL_TASKS = ['semseg', 'human_parts', 'normals', 'edge', 'sal', 'depth']


def get_output_num(task, dataname):
    if dataname == 'pascalcontext':
        return PASCAL_OUT_CHANNELS[task]
    elif dataname == 'nyud':
        return NYUD_OUT_CHANNELS[task]
    elif dataname == 'cityscapes':
        return CITYSCAPES_OUT_CHANNELS[task]
    else:
        raise NotImplementedError