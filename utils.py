import os

import numpy as np
import torch
import torch.nn.functional as F

from datasets.custom_transforms import get_transformations
from datasets.utils.configs import NUM_TRAIN_IMAGES, TRAIN_SCALE, TEST_SCALE


def get_st_config(dataset_configs):
    st_configs = {}
    for data_config in dataset_configs:
        dataname = data_config['dataname']
        train_transforms = get_transformations(TRAIN_SCALE[dataname], train=True)
        val_transforms = get_transformations(TEST_SCALE[dataname], train=False)
        # number of models is defined in task_dict
        task_dict = data_config['task_dict']
        n_nets = sum(task_dict.values())
        print('Training %d single-task models on %s' % (n_nets, dataname))

        task_list = []
        for task_name in task_dict:
            task_list += [task_name] * task_dict[task_name]
        assert len(task_list) == n_nets

        # random partition of dataset
        idxs = np.random.permutation(NUM_TRAIN_IMAGES[dataname])
        if data_config['noniid']:
            # non-iid data partition
            n_shot = n_nets * (n_nets + 1) / 2
            one_shot = len(idxs) // n_shot
            batch_idxs = []
            p = 0
            for i in range(n_nets):
                batch_idxs.append(idxs[int(p * one_shot):int((p + i + 1) * one_shot)])
                p += (i + 1)
        else:
            batch_idxs = np.array_split(idxs, n_nets)
        net_task_dataidx_map = [{'task_list': [task_list[i]], 'dataidx': batch_idxs[i]} for i in range(n_nets)]

        st_configs[dataname] = data_config  # defined in yml
        st_configs[dataname]['n_nets'] = n_nets
        st_configs[dataname]['train_transforms'] = train_transforms
        st_configs[dataname]['val_transforms'] = val_transforms
        st_configs[dataname]['net_task_dataidx_map'] = net_task_dataidx_map

    return st_configs


def get_mt_config(dataset_configs):
    mt_configs = {}
    for data_config in dataset_configs:
        dataname = data_config['dataname']
        train_transforms = get_transformations(TRAIN_SCALE[dataname], train=True)
        val_transforms = get_transformations(TEST_SCALE[dataname], train=False)

        # number of models is defined in client_num
        n_nets = data_config['client_num']
        print('Training %d multi-task models on %s' % (n_nets, dataname))

        task_dict = data_config['task_dict']
        task_list = []
        for task_name in task_dict:
            task_list += [task_name] * task_dict[task_name]

        # random partition of dataset
        idxs = np.random.permutation(NUM_TRAIN_IMAGES[dataname])
        if data_config['noniid']:
            # non-iid data partition
            n_shot = n_nets * (n_nets + 1) / 2
            one_shot = len(idxs) // n_shot
            batch_idxs = []
            p = 0
            for i in range(n_nets):
                batch_idxs.append(idxs[int(p * one_shot):int((p + i + 1) * one_shot)])
                p += (i + 1)
        else:
            batch_idxs = np.array_split(idxs, n_nets)
        net_task_dataidx_map = [{'task_list': task_list, 'dataidx': batch_idxs[i]} for i in range(n_nets)]

        mt_configs[dataname] = data_config  # defined in yml
        mt_configs[dataname]['n_nets'] = n_nets
        mt_configs[dataname]['train_transforms'] = train_transforms
        mt_configs[dataname]['val_transforms'] = val_transforms
        mt_configs[dataname]['net_task_dataidx_map'] = net_task_dataidx_map

    return mt_configs


def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def create_results_dir(results_dir, exp_name):
    """
    Create required results directory if it does not exist
    :param results_dir: Directory to create subdirectory in
    :param exp_name: Name of experiment to be used in the directory created
    :return: Path of experiment directory and checkpoint directory
    """
    exp_dir = os.path.join(results_dir, exp_name)
    checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
    create_dir(results_dir)
    create_dir(exp_dir)
    create_dir(checkpoint_dir)
    return exp_dir, checkpoint_dir


def create_pred_dir(results_dir, exp_name, all_nets):
    """
    Create required prediction directory if it does not exist
    :param results_dir: Directory to create subdirectory in
    :param exp_name: Name of experiment to be used in the directory created
    :param tasks: Specified tasks
    :return: Path of checkpoint directory and prediction dictionary
    """
    exp_dir = os.path.join(results_dir, exp_name)
    checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
    pred_dir = os.path.join(exp_dir, 'predictions')
    create_dir(pred_dir)

    for idx in all_nets:
        for task in all_nets[idx]['tasks']:
            task_dir = os.path.join(pred_dir, str(idx) + '_' + task)
            create_dir(task_dir)
            if task == 'edge':
                create_dir(os.path.join(task_dir, 'img'))

    return checkpoint_dir, pred_dir


class RunningMeter(object):

    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_loss_metric(loss_meter, tasks, prefix, idx, mt=False):
    """
    Get loss statistics
    :param loss_meter: Loss meter
    :param tasks: List of tasks
    :param prefix: Prefix for the loss, train or val
    :return: loss statistics
    """
    if mt:
        statistics = {prefix + '/' + str(idx) + '_loss_sum': 0.0}
    else:
        statistics = {}
    for task in tasks:
        if mt:
            statistics[prefix + '/' + str(idx) + '_loss_sum'] += loss_meter[task].avg
        statistics[prefix + '/' + str(idx) + '_' + task] = loss_meter[task].avg
        loss_meter[task].reset()

    return statistics


def to_cuda(batch):
    if type(batch) is dict:
        out = {}
        for k, v in batch.items():
            if k == 'meta':
                out[k] = v
            else:
                out[k] = to_cuda(v)
        return out
    elif type(batch) is torch.Tensor:
        return batch.cuda(non_blocking=True)
    elif type(batch) is list:
        return [to_cuda(v) for v in batch]
    else:
        return batch


def get_output(output, task):
    if task == 'normals':
        output = output.permute(0, 2, 3, 1)
        output = (F.normalize(output, p=2, dim=3) + 1.0) * 255 / 2.0

    elif task in {'semseg', 'human_parts'}:
        output = output.permute(0, 2, 3, 1)
        _, output = torch.max(output, dim=3)

    elif task in {'edge'}:
        output = output.permute(0, 2, 3, 1)
        output = torch.sigmoid(output).squeeze(-1) * 255

    elif task in {'sal'}:
        output = output.permute(0, 2, 3, 1)
        output = F.softmax(output, dim=3)[:, :, :, 1] * 255

    elif task in {'depth'}:
        output.clamp_(min=0.)
        output = output.permute(0, 2, 3, 1).squeeze(-1)

    else:
        raise ValueError('Select one of the valid tasks')

    return output


def move_ckpt(ckpt_dict, device):
    for i in ckpt_dict.keys():
        for key in ckpt_dict[i].keys():
            ckpt_dict[i][key] = ckpt_dict[i][key].to(device)

    return ckpt_dict

# task conditional prompt
def get_task_latent(tasks, dataname, dim_latent, device):
    if dataname == 'pascalcontext':
        unit_dim = dim_latent // 5
    elif dataname == 'nyud':
        unit_dim = dim_latent // 4
    else:
        raise ValueError()

    task_latent_z = {}
    for task in tasks:
        z = torch.zeros((1, dim_latent), dtype=torch.float32, device=device)
        if task == 'semseg':
            z[:, :1 * unit_dim] = 1
        elif task in {'human_parts', 'depth'}:
            z[:, 1 * unit_dim:2 * unit_dim] = 1
        elif task == 'normals':
            z[:, 2 * unit_dim:3 * unit_dim] = 1
        elif task == 'edge':
            z[:, 3 * unit_dim:4 * unit_dim] = 1
        elif task == 'sal':
            z[:, 4 * unit_dim:5 * unit_dim] = 1
        task_latent_z[task] = z

    return task_latent_z

# domain-task conditional prompt
def get_task_latent_dtc(tasks, dataname, dim_latent, device):
    unit_dim = dim_latent // 8

    task_latent_z = {}
    for task in tasks:
        z = torch.zeros((1, dim_latent), dtype=torch.float32, device=device)
        # encoding for dataset
        if dataname == 'pascalcontext':
            z[:, :(unit_dim // 2)] = 1
        elif dataname == 'nyud':
            z[:, (unit_dim // 2):unit_dim] = 1
        
        # encoding for task
        if task == 'normals':
            z[:, 1 * unit_dim:2 * unit_dim] = 1
        elif task == 'edge':
            z[:, 2 * unit_dim:3 * unit_dim] = 1
        elif task == 'semseg':
            if dataname == 'pascalcontext':
                z[:, 3 * unit_dim:4 * unit_dim] = 1
            elif dataname == 'nyud':
                z[:, 4 * unit_dim:5 * unit_dim] = 1
        elif task == 'human_parts':
            z[:, 5 * unit_dim:6 * unit_dim] = 1
        elif task == 'sal':
            z[:, 6 * unit_dim:7 * unit_dim] = 1
        elif task == 'depth':
            z[:, 7 * unit_dim:8 * unit_dim] = 1

        task_latent_z[task] = z

    return task_latent_z

def get_task_latent_Mode(tasks, dataname, dim_latent, device, Mode='tc'):
    task_latent_z = {}
    if Mode == 'dtc':
        task_latent_z = get_task_latent_dtc(tasks, dataname, dim_latent, device)
    elif Mode == 'tc':
        task_latent_z = get_task_latent(tasks, dataname, dim_latent, device)
    elif Mode == 'nc':
        for task in tasks:
            z = torch.zeros((1, dim_latent), dtype=torch.float32, device=device)
            task_latent_z[task] = z
    return task_latent_z

def flatten_model(model):
    state_dict = model.state_dict()
    keys = state_dict.keys()
    W = [state_dict[key].flatten() for key in keys]
    return torch.cat(W)

def flatten_mdmodel(model):
    state_dict = model.module.state_dict()
    keys = [name for name,_ in model.module.named_parameters()]
    enc_keys = list(filter(lambda x: 'encoder' in x, keys))
    dec_keys = list(filter(lambda x: 'decoder' in x and 'conv_last' not in x, keys))
    all_keys = enc_keys + dec_keys
    W = [state_dict[key].flatten() for key in all_keys]
    return torch.cat(W)