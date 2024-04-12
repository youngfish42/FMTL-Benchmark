import copy

import numpy as np
import torch
from scipy.optimize import minimize
from sklearn.cluster import AgglomerativeClustering
from utils import move_ckpt
import math


def get_encoder_keys(all_keys):
    return list(filter(lambda x: 'encoder' in x, all_keys))


def get_decoder_keys(all_keys):
    return list(filter(lambda x: 'decoder' in x, all_keys))


def get_decoder_keys_stmd(all_keys):
    return list(filter(lambda x: 'decoder' in x and 'last' not in x, all_keys))


def get_model_soup(param_dict_list):
    soup_param_dict = {}
    layers = param_dict_list[0].keys()
    for layer in layers:
        soup_param_dict[layer] = torch.mean(torch.stack(
            [param_dict_list[i][layer] for i in range(len(param_dict_list))]),
                                            dim=0)

    return soup_param_dict


def get_delta_dict_list(param_dict_list, last_param_dict_list):
    # a list of length K, each element is a dict of delta parameters
    delta_dict_list = []
    layers = param_dict_list[0].keys()
    for i in range(len(param_dict_list)):
        delta_dict_list.append({})
        for layer in layers:
            delta_dict_list[i][layer] = param_dict_list[i][layer] - \
                last_param_dict_list[i][layer]

    return delta_dict_list


def get_encoder_params(all_nets, ckpt):
    # encoder_param_list: a list of length n_st, each element is a dict of encoder parameters
    all_name_keys = [name for name, _ in all_nets[0]['model'].module.named_parameters()]
    # all_name_keys = [name for name, _ in all_nets[0]['model'].named_parameters()]
    encoder_keys = get_encoder_keys(all_name_keys)
    encoder_param_dict_list = []
    shapes = []

    for model_idx in range(len(ckpt)):
        param_dict = {}
        for key in encoder_keys:
            param_dict[key] = ckpt[model_idx][key]
            if model_idx == 0:
                shapes.append(ckpt[model_idx][key].shape)
        encoder_param_dict_list.append(param_dict)

    return encoder_param_dict_list, encoder_keys, shapes


def get_decoder_params(all_nets, ckpt):
    # decoder_param_list: a list of length n_st, each element is a dict of decoder parameters
    all_name_keys = [name for name, _ in all_nets[0]['model'].module.named_parameters()]
    # all_name_keys = [name for name, _ in all_nets[0]['model'].named_parameters()]
    decoder_keys = get_decoder_keys(all_name_keys)
    decoder_param_dict_list = []
    shapes = []

    for model_idx in range(len(ckpt)):
        param_dict = {}
        for key in decoder_keys:
            param_dict[key] = ckpt[model_idx][key]
            if model_idx == 0:
                shapes.append(ckpt[model_idx][key].shape)
        decoder_param_dict_list.append(param_dict)

    return decoder_param_dict_list, decoder_keys, shapes


def get_decoder_params_stmd(all_nets, ckpt):
    # decoder_param_list: a list of length n_st, each element is a dict of decoder parameters
    decoder_keys = []
    layers = []
    shapes = []

    for idx in range(len(ckpt)):
        all_name_keys = [key for key, _ in all_nets[idx]['model'].module.named_parameters()]
        decoder_keys += get_decoder_keys_stmd(all_name_keys)
    decoder_keys = list(set(decoder_keys))

    decoder_param_dict_list = []
    decoders_prefix = []
    # st client decoders
    for model_idx in range(len(ckpt)):
        assert len(all_nets[model_idx]['tasks']) == 1
        param_dict = {}
        for key in decoder_keys:
            if key in ckpt[model_idx].keys():
                # key=prefix+'.'+layer
                prefix = key.split('.', 2)[0] + '.' + \
                    key.split('.', 2)[1]  # decoders.task
                layer = key.split('.', 2)[2]
                param_dict[layer] = ckpt[model_idx][key]

                if model_idx == 0:
                    layers.append(layer)
                    shapes.append(ckpt[0][key].shape)

        decoders_prefix.append(prefix)
        decoder_param_dict_list.append(param_dict)

    return decoder_param_dict_list, decoders_prefix, decoder_keys, layers, shapes


def get_all_params(all_nets, ckpt):
    # decoder_param_list: a list of length n_st, each element is a dict of decoder parameters
    all_name_keys = [name for name, _ in all_nets[0]['model'].module.named_parameters()]
    # all_name_keys = [name for name, _ in all_nets[0]['model'].named_parameters()]
    param_dict_list = []

    for model_idx in range(len(ckpt)):
        param_dict = {}
        for key in all_name_keys:
            param_dict[key] = ckpt[model_idx][key]
        param_dict_list.append(param_dict)

    return param_dict_list, all_name_keys


def get_pcgrad_delta_all(flatten_delta_list):
    N = len(flatten_delta_list)
    # norm flatten_delta_list
    for i in range(N):
        flatten_delta_list[i] /= (flatten_delta_list[i].norm() + 1e-8)
    PC = copy.deepcopy(flatten_delta_list)

    for i in range(N):
        idx_list = list(range(N))
        idx_list.remove(i)
        # random shuffle
        np.random.shuffle(idx_list)
        for j in idx_list:
            cth = torch.dot(PC[i], flatten_delta_list[j])
            if cth < 0:
                PC[i] -= cth * flatten_delta_list[j] / \
                    ((flatten_delta_list[j].norm())**2+1e-8)
    final_update = torch.stack([PC[model_idx] for model_idx in range(N)]).mean(dim=0)

    return final_update


def get_cagrad_delta_all(flatten_delta_list, alpha=0.4, rescale=1):
    N = len(flatten_delta_list)
    grads = torch.stack(flatten_delta_list).t()  # [d , N]
    GG = grads.t().mm(grads).cpu()  # [N, N]
    g0_norm = (GG.mean() + 1e-8).sqrt()

    x_start = np.ones(N) / N
    bnds = tuple((0, 1) for x in x_start)
    cons = ({'type': 'eq', 'fun': lambda x: 1 - sum(x)})
    A = GG.numpy()
    b = x_start.copy()
    c = (alpha * g0_norm + 1e-8).item()

    def objfn(x):
        return (x.reshape(1, -1).dot(A).dot(b.reshape(-1, 1)) +
                c * np.sqrt(x.reshape(1, -1).dot(A).dot(x.reshape(-1, 1)) + 1e-8)).sum()

    res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
    ww = torch.Tensor(res.x).to(grads.device)
    gw = (grads * ww.reshape(1, -1)).sum(1)
    gw_norm = gw.norm()
    lmbda = c / (gw_norm + 1e-8)
    g = grads.mean(1) + lmbda * gw
    if rescale == 0:
        final_update = g
    elif rescale == 1:
        final_update = g / (1 + alpha**2)
    else:
        final_update = g / (1 + alpha)

    return final_update


def flatten_param(param_dict_list, keys):
    flatten_list = [torch.cat([param_dict_list[idx][k].flatten() for k in keys]) for idx in range(len(param_dict_list))]
    assert len(flatten_list[0].shape) == 1

    return flatten_list


def unflatten_param(flatten_param, shapes, keys):
    param_dict = {}
    start = 0
    for k, shape in zip(keys, shapes):
        end = start + np.prod(shape)
        param_dict[k] = flatten_param[start:end].reshape(shape)
        start = end
    return param_dict


def unflatten_param_list(flatten_list, shapes, keys):
    param_dict_list = []
    for model_idx in range(len(flatten_list)):
        start = 0
        param_dict_list.append({})
        for key, shape in zip(keys, shapes):
            end = start + np.prod(shape)
            param_dict_list[model_idx][key] = flatten_list[model_idx][start:end].reshape(shape)
            start = end

    return param_dict_list


def get_grouping_score(encoder_param_list, encoder_keys, cluster_num):
    model_soup = get_model_soup(encoder_param_list)

    delta_list = []
    for key in encoder_keys:
        temp_delta = torch.stack([ckpt[key] for ckpt in encoder_param_list], dim=0) - model_soup[key]
        delta_list.append(temp_delta.reshape([len(temp_delta), -1]))

    delta = torch.cat(delta_list, dim=1)
    clustering = AgglomerativeClustering(n_clusters=cluster_num, metric='cosine', linkage='average').fit(delta.cpu())
    # print(clustering.labels_)
    cluster_results = torch.tensor(clustering.labels_).cuda()
    scores = torch.eq(cluster_results.view(-1, 1), cluster_results.view(1, -1)).float()
    scores = scores / scores.sum(dim=1, keepdim=True)

    return scores


def aggregate_module(update_ckpt, param_list, keys, shapes, last_param_list=None, agg='none', alphak=1.0, sigma=1.0):
    N = len(param_list)

    if agg in ['fedavg', 'fedprox', 'ditto']:
        new_param = get_model_soup(param_list)

        for model_idx in range(N):
            for key in keys:
                update_ckpt[model_idx][key] = new_param[key]

    elif agg in ['pcgrad', 'cagrad']:
        assert last_param_list is not None
        delta_list = get_delta_dict_list(param_list, last_param_list)
        # flatten
        flatten_delta = flatten_param(delta_list, keys)
        del delta_list, param_list

        # solve for aggregated conflict-averse delta
        if agg in ['pcgrad']:
            flatten_delta_update = get_pcgrad_delta_all(flatten_delta)
        elif agg in ['cagrad']:
            flatten_delta_update = get_cagrad_delta_all(flatten_delta)
        else:
            raise NotImplementedError

        delta_update = unflatten_param(flatten_delta_update, shapes, keys)

        for model_idx in range(N):
            for key in keys:
                update_ckpt[model_idx][key] = last_param_list[model_idx][key] + delta_update[key]

    elif agg in ['fedamp']:
        for i, ow in enumerate(param_list):
            mu = copy.deepcopy(param_list[0])
            for param in mu.values():
                param.zero_()

            coef = torch.zeros(N)
            for j, mw in enumerate(param_list):
                if i != j:
                    weights_i_list = []
                    weights_j_list = []
                    for key in keys:
                        weights_i_list.append(ow[key].view(-1))
                        weights_j_list.append(mw[key].view(-1))
                    weights_i = torch.cat(weights_i_list, dim=0)
                    weights_j = torch.cat(weights_j_list, dim=0)
                    sub = (weights_i - weights_j).view(-1)
                    sub = torch.dot(sub, sub)
                    coef[j] = alphak * math.exp(-sub / sigma) / sigma

                else:
                    coef[j] = 0
            coef_self = 1 - torch.sum(coef)

            for j, mw in enumerate(param_list):
                for key in keys:
                    mu[key] += coef[j] * mw[key]

            for key in keys:
                update_ckpt[i][key] = (mu[key] + coef_self * param_list[i][key]).clone()

    elif agg in ['manytask']:
        scores = get_grouping_score(param_list, keys, cluster_num=2)
        for key in keys:
            temp_weight = torch.stack([param_list[i][key] for i in range(N)], dim=0)
            reshaped_weights = temp_weight.reshape([N, -1])
            orig_shape = temp_weight.shape
            scores = scores.to(reshaped_weights.device)
            reweighted_weights = torch.matmul(scores, reshaped_weights).reshape(orig_shape)

            for model_idx in range(N):
                update_ckpt[model_idx][key] = reweighted_weights[model_idx]

    else:
        raise NotImplementedError


def aggregate(all_nets,
              save_ckpt,
              last_ckpt,
              encoder_agg='none',
              decoder_agg='none',
              alphak=1.0,
              sigma=1.0,
              hypernet=None) -> dict:

    assert len(all_nets) == len(save_ckpt)
    N = len(all_nets)

    if encoder_agg == 'none' and decoder_agg == 'none':
        return  # no aggregation

    update_ckpt = copy.deepcopy(save_ckpt)  # store updated parameters

    if not encoder_agg == 'none':
        # get encoder parameter list
        encoder_param_list, encoder_keys, enc_shapes = get_encoder_params(all_nets, save_ckpt)
        if encoder_agg in ['pcgrad', 'cagrad', 'fedhca2']:
            last_encoder_param_list, _, _ = get_encoder_params(all_nets, last_ckpt)
        else:
            last_encoder_param_list = None

        if encoder_agg == 'fedhca2':
            assert hypernet['enc_model'] is not None
            encoder_delta_list = get_delta_dict_list(encoder_param_list, last_encoder_param_list)

            # flatten
            del encoder_param_list
            flatten_last_encoder = flatten_param(last_encoder_param_list, encoder_keys)
            del last_encoder_param_list
            flatten_encoder_delta = flatten_param(encoder_delta_list, encoder_keys)
            del encoder_delta_list

            # delta balancing
            flatten_delta_update = get_cagrad_delta_all(flatten_encoder_delta)  # flattened tensor

            # update
            flatten_new_encoder = hypernet['enc_model'](flatten_last_encoder, flatten_encoder_delta,
                                                        flatten_delta_update)
            # record output of hypernetwork for backprop
            hypernet['last_enc_output'] = flatten_new_encoder

            del flatten_last_encoder, flatten_encoder_delta, flatten_delta_update

            new_encoder_param_list = unflatten_param_list(flatten_new_encoder, enc_shapes, encoder_keys)

            for model_idx in range(N):
                for key in encoder_keys:
                    update_ckpt[model_idx][key] = new_encoder_param_list[model_idx][key]

            del new_encoder_param_list
        else:
            aggregate_module(update_ckpt, encoder_param_list, encoder_keys, enc_shapes, last_encoder_param_list,
                             encoder_agg, alphak, sigma)

            del encoder_param_list, last_encoder_param_list

    if decoder_agg not in ['none', 'manytask']:
        # get decoder parameter list
        decoder_param_list, decoder_keys, dec_shapes = get_decoder_params(all_nets, save_ckpt)
        if decoder_agg in ['pcgrad', 'cagrad', 'fedhca2']:
            last_decoder_param_list, _, _ = get_decoder_params(all_nets, last_ckpt)
        else:
            last_decoder_param_list = None

        if decoder_agg == 'fedhca2':
            assert hypernet['dec_model'] is not None
            decoder_delta_list = get_delta_dict_list(decoder_param_list, last_decoder_param_list)
            del decoder_param_list

            new_decoder_param_list = hypernet['dec_model'](last_decoder_param_list, decoder_delta_list)
            # record output of hypernetwork for backprop
            hypernet['last_dec_output'] = new_decoder_param_list
            del decoder_delta_list, last_decoder_param_list

            for model_idx in range(N):
                for key in decoder_keys:
                    update_ckpt[model_idx][key] = new_decoder_param_list[model_idx][key]

            del new_decoder_param_list
        else:
            aggregate_module(update_ckpt, decoder_param_list, decoder_keys, dec_shapes, last_decoder_param_list,
                             decoder_agg, alphak, sigma)

            del decoder_param_list, last_decoder_param_list

        # decoder_param_list, decoders_prefix, decoder_keys, dec_layers, dec_shapes = get_decoder_params_stmd(
        #     all_nets, save_ckpt)
        # new_decoder_param = get_model_soup(decoder_param_list)
        # for i, prefix in enumerate(decoders_prefix):
        #     for layer in dec_layers:
        #         update_ckpt[i][prefix + '.' + layer] = new_decoder_param[layer]

        # del decoder_param_list, new_decoder_param

    # update all models
    update_ckpt = move_ckpt(update_ckpt, 'cuda')
    if encoder_agg == 'fedamp':
        for model_idx in range(N):
            all_nets[model_idx]['p_model'].module.load_state_dict(update_ckpt[model_idx])
    else:
        for model_idx in range(N):
            all_nets[model_idx]['model'].module.load_state_dict(update_ckpt[model_idx])

    del update_ckpt


def update_hypernetwork(all_nets, hypernet, save_ckpt, last_ckpt):
    if 'enc_model' in hypernet.keys():
        # get encoder parameter list and prefix
        encoder_param_list, encoder_keys, enc_shapes = get_encoder_params(all_nets, save_ckpt)
        last_encoder_param_list, _, _ = get_encoder_params(all_nets, last_ckpt)

        # calculate difference between current and last encoder parameters
        diff_list = get_delta_dict_list(last_encoder_param_list, encoder_param_list)
        flatten_diff = flatten_param(diff_list, encoder_keys)

        # update hypernetwork
        hypernet['enc_model'].train()
        optimizer = hypernet['enc_optimizer']
        optimizer.zero_grad()

        torch.autograd.backward(hypernet['last_enc_output'], flatten_diff, retain_graph=True)

        optimizer.step()

    if 'dec_model' in hypernet.keys():
        # get decoder parameter list and prefix
        decoder_param_list, decoder_keys, dec_shapes = get_decoder_params(all_nets, save_ckpt)
        last_decoder_param_list, _, _ = get_decoder_params(all_nets, last_ckpt)

        # calculate difference between current and last decoder parameters
        diff_list = get_delta_dict_list(last_decoder_param_list, decoder_param_list)

        # update hypernetwork
        hypernet['dec_model'].train()
        optimizer = hypernet['dec_optimizer']
        optimizer.zero_grad()

        for i in range(len(decoder_param_list)):
            # construct dict of parameters into list
            last_output = list(map(lambda x: hypernet['last_dec_output'][i][x], decoder_keys))
            diff_param = list(map(lambda x: diff_list[i][x], decoder_keys))
            torch.autograd.backward(last_output, diff_param, retain_graph=True)

        optimizer.step()
