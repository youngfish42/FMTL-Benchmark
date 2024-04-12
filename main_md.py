import argparse
import copy
import datetime
import math
import os
import shutil
import time

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import wandb
import yaml
from carbontracker.tracker import CarbonTracker
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from aggregate_all import aggregate
from aggregate_md import aggregate_md, update_hypernetwork
from datasets.custom_dataset import get_dataloader, get_dataset
from evaluation.evaluate_utils import PerformanceMeter
from losses import get_criterion
from models.hypernet import HyperAttention, HyperWeightALL
from models.model import MD_model
from utils import (RunningMeter, create_results_dir, flatten_mdmodel, get_loss_metric, get_mt_config, get_output,
                   get_st_config, move_ckpt, to_cuda)

wandb_name = "FMTL-Bench"


def local_train(tasks, train_dl, local_epochs, model, optimizer, scheduler, p_model, p_optimizer, p_scheduler,
                criterion, scaler, train_loss, cr, idx, local_rank, lamda, mu, agg, alphak, omega, fp16, W_glob,
                **args):
    model.train()
    p_model.train()
    # random shuffle tasks
    order = np.arange(len(tasks))
    np.random.shuffle(order)
    local_params = copy.deepcopy(list(model.module.parameters()))

    if agg == 'fedmtl':
        W_glob = W_glob.cuda()

    for epoch in range(local_epochs):
        train_dl.sampler.set_epoch(cr * local_epochs + epoch)
        for batch in tqdm(train_dl,
                          desc='CR: %d Local Epoch: %d Net %d Task: %s' % (cr, epoch, idx, tasks),
                          disable=(local_rank != 0)):
            optimizer.zero_grad()
            batch = to_cuda(batch)
            images = batch['image']
            batch_size = images.shape[0]

            if agg == 'fedamp':
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=fp16):
                    outputs = model(images)
                    loss_dict = criterion(outputs, batch, tasks)
                    gm = torch.cat([p.data.view(-1) for p in model.parameters()], dim=0)
                    pm = torch.cat([p.data.view(-1) for p in p_model.parameters()], dim=0)
                    loss_dict['total'] += 0.5 * lamda / alphak * torch.norm(gm - pm, p=2)

                scaler.scale(loss_dict['total']).backward()
                scaler.step(optimizer)
                scaler.update()
                for task in tasks:
                    loss_value = loss_dict[task].detach().item()
                    train_loss[task].update(loss_value / batch_size, batch_size)

            elif agg == 'fedprox':
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=fp16):
                    outputs = model(images)
                    loss_dict = criterion(outputs, batch, tasks)
                    prox = 0
                    for param, localweight in zip(model.module.parameters(), local_params):
                        prox += torch.norm(param.data - localweight.data, 2)**2
                    loss_dict['total'] += (mu / 2) * prox
                for task in tasks:
                    loss_value = loss_dict[task].detach().item()
                    train_loss[task].update(loss_value / batch_size, batch_size)

                scaler.scale(loss_dict['total']).backward()
                scaler.step(optimizer)
                scaler.update()

            elif agg == 'diito':
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=fp16):
                    outputs = p_model(images)
                    loss_dict = criterion(outputs, batch, tasks)
                    prox = 0
                    for param, localweight in zip(p_model.module.parameters(), local_params):
                        prox += torch.norm(param.data - localweight.data, 2)**2
                    loss_dict['total'] += (mu / 2) * prox
                for task in tasks:
                    loss_value = loss_dict[task].detach().item()
                    train_loss[task].update(loss_value / batch_size, batch_size)

                scaler.scale(loss_dict['total']).backward()
                scaler.step(p_optimizer)
                scaler.update()

            elif agg == 'fedmtl':
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=fp16):
                    outputs = model(images)
                    loss_dict = criterion(outputs, batch, tasks)
                    W_glob[:, idx] = flatten_mdmodel(model)
                    loss_reg = 0
                    loss_reg += W_glob.norm()**2
                    loss_reg += torch.sum(torch.sum((W_glob * omega), 1)**2)
                    f = (int)(math.log10(W_glob.shape[0]) + 1) + 1
                    loss_reg *= 10**(-f)
                    loss_dict['total'] += loss_reg
                for task in tasks:
                    loss_value = loss_dict[task].detach().item()
                    train_loss[task].update(loss_value / batch_size, batch_size)

                scaler.scale(loss_dict['total']).backward()
                scaler.step(optimizer)
                scaler.update()

            else:
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=fp16):
                    outputs = model(images)
                    loss_dict = criterion(outputs, batch, tasks)

                for task in tasks:
                    loss_value = loss_dict[task].detach().item()
                    train_loss[task].update(loss_value / batch_size, batch_size)

                scaler.scale(loss_dict['total']).backward()
                scaler.step(optimizer)
                scaler.update()
        if agg != 'ditto':
            scheduler.step(cr * local_epochs + epoch)
        else:
            p_scheduler.step(cr * local_epochs + epoch)
    if agg == 'ditto':
        for epoch in range(local_epochs):
            train_dl.sampler.set_epoch(cr * local_epochs + epoch)
            for batch in tqdm(train_dl,
                              desc='CR: %d Local Epoch: %d Net %d Task: %s' % (cr, epoch, idx, tasks),
                              disable=(local_rank != 0)):
                optimizer.zero_grad()
                batch = to_cuda(batch)
                images = batch['image']
                batch_size = images.shape[0]
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=fp16):
                    outputs = model(images)
                    loss_dict = criterion(outputs, batch, tasks)
                    prox = 0
                    for param, localweight in zip(model.module.parameters(), local_params):
                        prox += torch.norm(param.data - localweight.data, 2)**2
                    loss_dict['total'] += (mu / 2) * prox
                for task in tasks:
                    loss_value = loss_dict[task].detach().item()
                    train_loss[task].update(loss_value / batch_size, batch_size)

                scaler.scale(loss_dict['total']).backward()
                scaler.step(optimizer)
                scaler.update()
            scheduler.step(cr * local_epochs + epoch)


def eval_metric(tasks, dataname, pval_dl, gval_dl, model, idx, **args):
    p_performance_meter = PerformanceMeter(dataname, tasks)
    g_performance_meter = PerformanceMeter(dataname, tasks)

    model.eval()
    with torch.no_grad():
        for batch in tqdm(pval_dl, desc='Evaluating Net %d on p' % (idx)):
            batch = to_cuda(batch)
            images = batch['image']
            outputs = model.module(images)
            p_performance_meter.update({t: get_output(outputs[t], t) for t in tasks}, batch)

        for batch in tqdm(gval_dl, desc='Evaluating Net %d on g' % (idx)):
            batch = to_cuda(batch)
            images = batch['image']
            outputs = model.module(images)
            g_performance_meter.update({t: get_output(outputs[t], t) for t in tasks}, batch)

    peval_results = p_performance_meter.get_score()
    geval_results = g_performance_meter.get_score()

    results_dict = {}
    for t in tasks:
        for key in peval_results[t]:
            results_dict['p_eval/' + str(idx) + '_' + t + '_' + key] = peval_results[t][key]
        for key in geval_results[t]:
            results_dict['g_eval/' + str(idx) + '_' + t + '_' + key] = geval_results[t][key]

    return results_dict


def main(all_nets, args, hypernet=None, local_rank=0):
    # get loss meters
    train_loss = {}
    val_loss = {}
    for idx in all_nets:
        train_loss[idx] = {}
        val_loss[idx] = {}
        for task in all_nets[idx]['tasks']:
            train_loss[idx][task] = RunningMeter()
            val_loss[idx][task] = RunningMeter()

    # save last_ckpt
    last_ckpt = {}
    for idx in all_nets:
        last_ckpt[idx] = copy.deepcopy(all_nets[idx]['model'].module.state_dict())
    if args.save_vram:
        last_ckpt = move_ckpt(last_ckpt, 'cpu')
    save_ckpt = copy.deepcopy(last_ckpt)

    #parameters used for FedMTL
    W_glob = None
    omega = None
    if args.encoder_agg in ['fedmtl']:
        num_join_clients = len(all_nets)
        dim = len(flatten_mdmodel(all_nets[0]['model']).cpu())
        W_glob = torch.zeros((dim, num_join_clients)).cuda()
        I = torch.ones((num_join_clients, num_join_clients))
        i = torch.ones((num_join_clients, 1))
        omega = (I - 1 / num_join_clients * i.mm(i.T))**2
        omega = torch.sqrt(omega[0][0])
        for idx in all_nets:
            W_glob[:, idx] = flatten_mdmodel(all_nets[idx]['model'])
        W_glob = W_glob.cpu()
    # using carbontracker to monitor the carbon footprint during training
    # https://github.com/lfwa/carbontracker/
    tracker = CarbonTracker(epochs=args.max_rounds,
                            epochs_before_pred=0,
                            monitor_epochs=args.max_rounds,
                            devices_by_pid=True,
                            verbose=2,
                            log_dir=args.exp_dir)

    for cr in range(args.max_rounds):
        tracker.epoch_start()
        # client update
        start_time = time.time()
        logs = {}
        for idx in all_nets:
            # train local models for local epochs
            W_per = copy.deepcopy(W_glob)
            local_train(train_loss=train_loss[idx],
                        cr=cr,
                        idx=idx,
                        local_rank=local_rank,
                        fp16=args.fp16,
                        lamda=args.lamda,
                        mu=args.mu,
                        agg=args.encoder_agg,
                        alphak=args.alphak,
                        omega=omega,
                        W_glob=W_per,
                        **all_nets[idx])

            train_stats = get_loss_metric(train_loss[idx], all_nets[idx]['tasks'], 'train', idx,
                                          (len(all_nets[idx]['tasks']) > 1))
            logs.update(train_stats)

        # update save_ckpt
        for idx in all_nets:
            save_ckpt[idx] = copy.deepcopy(all_nets[idx]['model'].module.state_dict())
        if args.save_vram:
            save_ckpt = move_ckpt(save_ckpt, 'cpu')

        # update hypernetwork
        if cr > 0:
            update_hypernetwork(all_nets, hypernet, save_ckpt, last_ckpt)

        # aggregate for traditional federated learning
        if args.encoder_agg in ['fedhca2']:
            aggregate_md(all_nets,
                         save_ckpt,
                         last_ckpt,
                         encoder_agg=args.encoder_agg,
                         decoder_agg=args.decoder_agg,
                         cagrad_c=args.cagrad_c,
                         hypernet=hypernet)

        elif args.encoder_agg not in ['fedamp', 'ditto', 'fedmtl']:
            aggregate(all_nets,
                      save_ckpt,
                      last_ckpt,
                      encoder_agg=args.encoder_agg,
                      decoder_agg=args.decoder_agg,
                      alphak=args.alphak,
                      sigma=args.sigma)
        # update last_ckpt
        for idx in all_nets:
            last_ckpt[idx] = copy.deepcopy(all_nets[idx]['model'].module.state_dict())
        if args.save_vram:
            last_ckpt = move_ckpt(last_ckpt, 'cpu')

        if args.encoder_agg in ['fedmtl']:
            num_join_clients = len(all_nets)
            dim = len(flatten_mdmodel(all_nets[0]['model']).cpu())
            W_glob = torch.zeros((dim, num_join_clients)).cuda()

            for idx in all_nets:
                W_glob[:, idx] = flatten_mdmodel(all_nets[idx]['model'])
            W_glob = W_glob.cpu()

            for idx in all_nets:
                last_ckpt[idx] = copy.deepcopy(all_nets[idx]['model'].module.state_dict())
            if args.save_vram:
                last_ckpt = move_ckpt(last_ckpt, 'cpu')
        end_time = time.time()

        if local_rank == 0:
            print("CR %d finishs, Time: %.1fs." % (cr, end_time - start_time))

            if (cr + 1) == args.max_rounds or (cr + 1) % args.eval_freq == 0:
                print('Validation at CR %d.' % (cr))
                # Evaluation on metrics
                val_logs = {}

                #diito's evaluation has to exchange to personalized model
                for idx in all_nets:
                    if args.encoder_agg == 'ditto':
                        temp_model = all_nets[idx]['model']
                        all_nets[idx]['model'] = all_nets[idx]['p_model']
                        all_nets[idx]['p_model'] = temp_model

                    res = eval_metric(idx=idx, **all_nets[idx])
                    val_logs.update(res)
                wandb.log({**logs, **val_logs})

                # save checkpoint
                save_ckpt_temp = {}
                for idx in all_nets:
                    save_ckpt_temp[idx] = copy.deepcopy(all_nets[idx]['model'].module.state_dict())
                torch.save(save_ckpt_temp, os.path.join(args.checkpoint_dir, 'checkpoint.pth'))
                print('Checkpoint saved.')
                del save_ckpt_temp

                #exchange back
                if args.encoder_agg == 'ditto':
                    for idx in all_nets:
                        temp_model = all_nets[idx]['model']
                        all_nets[idx]['model'] = all_nets[idx]['p_model']
                        all_nets[idx]['p_model'] = temp_model
            else:
                wandb.log(logs)

        # aggregate for personalized federated learning
        if args.encoder_agg in ['fedamp', 'ditto']:
            aggregate(all_nets,
                      save_ckpt,
                      last_ckpt,
                      encoder_agg=args.encoder_agg,
                      decoder_agg=args.decoder_agg,
                      alphak=args.alphak,
                      sigma=args.sigma)

            # update last_ckpt
            for idx in all_nets:
                last_ckpt[idx] = copy.deepcopy(all_nets[idx]['model'].module.state_dict())
            if args.save_vram:
                last_ckpt = move_ckpt(last_ckpt, 'cpu')
        tracker.epoch_end()
    if local_rank == 0:
        print('Training finished.')


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_clients(client_configs, model_config, all_nets, net_idx, n_decoders, fp16=False):
    for dataname in client_configs:
        client_config = client_configs[dataname]
        net_task_dataidx_map, n_nets = client_config['net_task_dataidx_map'], client_config['n_nets']

        for idx in range(n_nets):
            task_list = net_task_dataidx_map[idx]['task_list']
            dataidxs = net_task_dataidx_map[idx]['dataidx']

            train_ds_local = get_dataset(dataname=dataname,
                                         train=True,
                                         tasks=task_list,
                                         transform=client_config['train_transforms'],
                                         dataidxs=dataidxs[:int(len(dataidxs) * 0.9)])
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds_local, drop_last=True)
            train_dl_local = get_dataloader(train=True,
                                            configs=client_config,
                                            dataset=train_ds_local,
                                            sampler=train_sampler)

            pval_ds_local = get_dataset(dataname=dataname,
                                        train=True,
                                        tasks=task_list,
                                        transform=client_config['val_transforms'],
                                        dataidxs=dataidxs[int(len(dataidxs) * 0.9):])
            pval_dl_local = get_dataloader(train=False, configs=client_config, dataset=pval_ds_local)

            gval_ds_local = get_dataset(dataname=dataname,
                                        train=False,
                                        tasks=task_list,
                                        transform=client_config['val_transforms'])
            gval_dl_local = get_dataloader(train=False, configs=client_config, dataset=gval_ds_local)

            model = MD_model(model_config['backbone'], task_list, dataname).cuda()
            model = DDP(model, device_ids=[local_rank])
            p_model = copy.deepcopy(model)
            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=float(client_config['lr']),
                                         weight_decay=float(client_config['weight_decay']))
            p_optimizer = torch.optim.Adam(p_model.parameters(),
                                           lr=float(client_config['lr']),
                                           weight_decay=float(client_config['weight_decay']))

            total_epochs = args.max_rounds * client_config['local_epochs']
            warmup_epochs = client_config['warmup_epochs']
            scheduler = CosineLRScheduler(optimizer=optimizer,
                                          t_initial=total_epochs - warmup_epochs,
                                          lr_min=1.25e-6,
                                          warmup_t=warmup_epochs,
                                          warmup_lr_init=1.25e-7,
                                          warmup_prefix=True)
            p_scheduler = CosineLRScheduler(optimizer=p_optimizer,
                                            t_initial=total_epochs - warmup_epochs,
                                            lr_min=1.25e-6,
                                            warmup_t=warmup_epochs,
                                            warmup_lr_init=1.25e-7,
                                            warmup_prefix=True)

            all_nets[net_idx]['tasks'] = task_list
            all_nets[net_idx]['dataname'] = dataname
            all_nets[net_idx]['train_dl'] = train_dl_local
            all_nets[net_idx]['pval_dl'] = pval_dl_local
            all_nets[net_idx]['gval_dl'] = gval_dl_local
            all_nets[net_idx]['local_epochs'] = client_config['local_epochs']
            all_nets[net_idx]['model'] = model
            all_nets[net_idx]['p_model'] = p_model
            all_nets[net_idx]['optimizer'] = optimizer
            all_nets[net_idx]['p_optimizer'] = p_optimizer
            all_nets[net_idx]['scheduler'] = scheduler
            all_nets[net_idx]['p_scheduler'] = p_scheduler
            all_nets[net_idx]['criterion'] = get_criterion(dataname, task_list).cuda()
            all_nets[net_idx]['scaler'] = torch.cuda.amp.GradScaler(enabled=fp16)
            net_idx += 1
            n_decoders += len(task_list)

    return net_idx, n_decoders


def str2bool(v):
    return v.lower() == 'true'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MTL-FL')
    parser.add_argument('--configs', type=str, default='./configs/nyud_mt_4c.yml')
    parser.add_argument('--exp', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--root_dir', type=str, default='./exp', help='root dir of results')
    parser.add_argument('--fp16', action='store_true', help='use fp16')
    parser.add_argument('--save_vram', action='store_true', help='save vram')

    parser.add_argument('--max_rounds', type=int, default=100)
    parser.add_argument('--eval_freq', type=int, default=4)

    # Notice: the aggregation method is only for the encoder of multi-decoder model
    parser.add_argument(
        '--encoder_agg',
        default='fedhca2',
        help='aggregation method for encoder',
        choices=['none', 'fedavg', 'fedamp', 'fedprox', 'ditto', 'manytask', 'pcgrad', 'cagrad', 'fedmtl', 'fedhca2'])
    parser.add_argument('--cagrad_c', type=float, default=0.4)
    parser.add_argument(
        '--decoder_agg',
        default='fedhca2',
        help='aggregation method for decoder',
        choices=['none', 'fedavg', 'fedamp', 'fedprox', 'ditto', 'manytask', 'pcgrad', 'cagrad', 'fedmtl', 'fedhca2'])
    # parameters for personalized federated learning
    parser.add_argument('--lamda', type=float, default=15)
    parser.add_argument('--mu', type=float, default=0.001)
    parser.add_argument('--sigma', type=float, default=1)
    parser.add_argument('--alphak', type=float, default=1)

    args = parser.parse_args()
    os.makedirs(args.root_dir, exist_ok=True)

    with open(args.configs, 'r') as stream:
        configs = yaml.safe_load(stream)

    # set seed and ddp
    set_seed(args.seed)
    dist.init_process_group('nccl', timeout=datetime.timedelta(0, 3600 * 2))
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    cudnn.benchmark = True
    cv2.setNumThreads(0)

    # setup logger and output folders
    args.exp_dir, args.checkpoint_dir = create_results_dir(args.root_dir, args.exp)
    if local_rank == 0:
        shutil.copy(args.configs, os.path.join(args.exp_dir, 'config.yml'))
        wandb.init(project=wandb_name, id=args.exp, name=args.exp, config={**configs, **vars(args)})
    dist.barrier()

    # get single-task and multi-task config
    n_all_nets = 0
    st_configs = {}
    mt_configs = {}

    if 'ST_Datasets' in configs:
        st_configs = get_st_config(configs['ST_Datasets'])
        n_all_nets += sum([st_configs[dataname]['n_nets'] for dataname in st_configs])

    if 'MT_Datasets' in configs:
        mt_configs = get_mt_config(configs['MT_Datasets'])
        n_all_nets += sum([mt_configs[dataname]['n_nets'] for dataname in mt_configs])

    # prepare all models
    all_nets = {idx: {} for idx in range(n_all_nets)}
    net_idx = 0
    n_decoders = 0

    # add clients
    net_idx, n_decoders = get_clients(st_configs, configs['Model'], all_nets, net_idx, n_decoders, args.fp16)
    net_idx, n_decoders = get_clients(mt_configs, configs['Model'], all_nets, net_idx, n_decoders, args.fp16)

    # setup hypernetwork
    hypernet = {}
    if args.encoder_agg in ['fedhca2']:
        model = HyperWeightALL(K=net_idx, init_gamma=0.1, norm=False)

        if args.save_vram:
            hypernet['enc_model'] = model
        else:
            hypernet['enc_model'] = DDP(model.cuda(), device_ids=[local_rank])
        hypernet['enc_optimizer'] = torch.optim.SGD(model.parameters(), **configs['Hypernetwork']['enc_opt'])

        dummy_decoder = all_nets[0]['model'].module.decoder
        model = HyperAttention(model=dummy_decoder, K=n_decoders, init_gamma=0.1, norm=False)

        if args.save_vram:
            hypernet['dec_model'] = model
        else:
            hypernet['dec_model'] = DDP(model.cuda(), device_ids=[local_rank])
        hypernet['dec_optimizer'] = torch.optim.SGD(model.parameters(), **configs['Hypernetwork']['dec_opt'])

    main(all_nets=all_nets, args=args, hypernet=hypernet, local_rank=local_rank)
    dist.destroy_process_group()
