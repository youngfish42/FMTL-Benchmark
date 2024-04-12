import argparse
import os

import torch
import yaml
from tqdm import tqdm

from datasets.custom_dataset import get_dataloader, get_dataset
from evaluation.evaluate_utils import PerformanceMeter, predict
from models.model import TSN, MD_model
from utils import (create_pred_dir, get_mt_config, get_output, get_st_config, get_task_latent_Mode, to_cuda)
from main import set_seed


def eval_metric(tasks, dataname, ptest_dl, gtest_dl, model, task_latent_z, idx, evaluate, save, pred_dir, tc, **args):
    p_performance_meter = PerformanceMeter(dataname, tasks)
    g_performance_meter = PerformanceMeter(dataname, tasks)
    if save:
        # save all tasks
        tasks_to_save = tasks
    else:
        # save only edge
        tasks_to_save = ['edge'] if 'edge' in tasks else []

    if not evaluate and len(tasks_to_save) == 0:
        return

    model.eval()
    with torch.no_grad():
        for dl, meter in zip([ptest_dl, gtest_dl], [p_performance_meter, g_performance_meter]):
            for batch in tqdm(dl, desc='Evaluating Net %d' % (idx)):
                batch = to_cuda(batch)
                images = batch['image']
                if tc:
                    outputs = {}
                    for task in tasks:
                        latent_z = task_latent_z[task].repeat(images.shape[0], 1)
                        outputs.update(model(images, latent_z, task, dataname))
                else:
                    outputs = model(images)

                if evaluate:
                    meter.update({t: get_output(outputs[t], t) for t in tasks}, batch)

                for task in tasks_to_save:
                    predict(dataname, batch['meta'], outputs, task, pred_dir, idx)

    if evaluate:
        # get evaluation results
        peval_results = p_performance_meter.get_score()
        geval_results = g_performance_meter.get_score()

        results_dict = {}
        for t in tasks:
            for key in peval_results[t]:
                results_dict['p_eval/' + str(idx) + '_' + t + '_' + key] = peval_results[t][key]
                results_dict['g_eval/' + str(idx) + '_' + t + '_' + key] = geval_results[t][key]

        return results_dict


def test(all_nets, args):
    test_results = {}
    for idx in all_nets:
        res = eval_metric(idx=idx,
                          evaluate=args.eval,
                          save=args.save,
                          pred_dir=args.pred_dir,
                          tc=args.tc,
                          **all_nets[idx])
        if args.eval:
            test_results.update({key: "%.4f" % res[key] for key in res})

    # log
    if args.eval:
        print(test_results)
        results_file = os.path.join(args.root_dir, args.exp, 'test_results.txt')
        with open(results_file, 'w') as f:
            f.write(str(test_results))


def get_clients(client_configs, model_config, all_nets, tc, Condition_Mode='tc'):
    net_idx = 0
    for dataname in client_configs:
        client_config = client_configs[dataname]
        net_task_dataidx_map, n_nets = client_config['net_task_dataidx_map'], client_config['n_nets']

        for idx in range(n_nets):
            task_list = net_task_dataidx_map[idx]['task_list']
            dataidxs = net_task_dataidx_map[idx]['dataidx']

            ptest_ds_local = get_dataset(dataname=dataname,
                                         train=True,
                                         tasks=task_list,
                                         transform=client_config['val_transforms'],
                                         dataidxs=dataidxs[int(len(dataidxs) * 0.9):])
            ptest_dl_local = get_dataloader(train=False, configs=client_config, dataset=ptest_ds_local)

            gtest_ds_local = get_dataset(dataname=dataname,
                                         train=False,
                                         tasks=task_list,
                                         transform=client_config['val_transforms'])
            gtest_dl_local = get_dataloader(train=False, configs=client_config, dataset=gtest_ds_local)

            if tc:
                model = TSN(**model_config).cuda()
            else:
                model = MD_model(task_list, dataname).cuda()

            all_nets[net_idx]['tasks'] = task_list
            all_nets[net_idx]['dataname'] = dataname
            all_nets[net_idx]['ptest_dl'] = ptest_dl_local
            all_nets[net_idx]['gtest_dl'] = gtest_dl_local
            all_nets[net_idx]['model'] = model
            all_nets[net_idx]['task_latent_z'] = get_task_latent_Mode(task_list, dataname, model_config['dim_latent'],
                                                                     torch.device('cuda'), Condition_Mode)

            net_idx += 1

    return net_idx


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FMTL-bench')
    parser.add_argument('--exp', type=str, required=True)
    parser.add_argument('--root_dir', type=str, default='./exp', help='root dir of results')
    parser.add_argument('--eval', action='store_true', help='evaluate models')
    parser.add_argument('--save', action='store_true', help='save predictions')
    parser.add_argument('--tc', action='store_true', help='task-conditioned model')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU id')

    # parameters for conditional prompt model
    parser.add_argument('--Mode', type=str, default='tc',help='type of conditional prompt',
                        choices=['dtc','tc','nc'])
    
    args = parser.parse_args()

    with open(os.path.join(args.root_dir, args.exp, 'config.yml'), 'r') as stream:
        configs = yaml.safe_load(stream)

    # set seed and ddp
    set_seed(0)
    torch.cuda.set_device(args.gpu_id)

    # get single-task and multi-task config
    n_all_nets = 0
    client_configs = {}
    if 'ST_Datasets' in configs:
        st_configs = get_st_config(configs['ST_Datasets'])
        n_all_nets += sum([st_configs[dataname]['n_nets'] for dataname in st_configs])
        client_configs.update(st_configs)

    if 'MT_Datasets' in configs:
        mt_configs = get_mt_config(configs['MT_Datasets'])
        n_all_nets += sum([mt_configs[dataname]['n_nets'] for dataname in mt_configs])
        client_configs.update(mt_configs)

    # prepare all models
    all_nets = {idx: {} for idx in range(n_all_nets)}

    # add clients
    net_idx = get_clients(client_configs, configs['Model'], all_nets, args.tc, args.Mode)

    # setup output folders
    args.checkpoint_dir, args.pred_dir = create_pred_dir(args.root_dir, args.exp, all_nets)

    # load model from checkpoint
    checkpoint_file = os.path.join(args.checkpoint_dir, 'checkpoint.pth')
    if not os.path.exists(checkpoint_file):
        raise ValueError('Checkpoint %s not found!' % (checkpoint_file))

    checkpoint = torch.load(checkpoint_file, map_location='cpu')
    for idx in all_nets:
        all_nets[idx]['model'].load_state_dict(checkpoint[idx])

    test(all_nets, args)
