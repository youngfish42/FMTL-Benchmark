import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class HyperAttention(nn.Module):

    def __init__(self, model, K, init_gamma=1, norm=True):
        super(HyperAttention, self).__init__()
        self.K = K
        self.norm = norm
        # get layer names
        self.layer_names = []
        for name, _ in model.named_parameters():
            if name.split('.')[0] in ['semseg','normals','depth','edge','human_parts','sal']:
                self.layer_names.append(".".join(name.split('.')[1:-1]))
            else:
                self.layer_names.append(".".join(name.split('.')[:-1]))
        self.layer_names = sorted(set(self.layer_names))
        self.gamma_names = [name.replace('.', '_') for name in self.layer_names]

        # define parameters
        self.gamma = nn.ParameterDict()
        for name in self.gamma_names:
            self.gamma[name] = nn.Parameter(torch.ones(K) * init_gamma)

    def forward(self, last_param_dict_list, delta_dict_list, avg_delta_list=None):
        new_param_dict_list = copy.deepcopy(last_param_dict_list)
        assert self.K == len(last_param_dict_list)  # number of models

        for name in self.layer_names:
            # cut gamma into [0, 1]
            layer_gamma = torch.clamp(self.gamma[name.replace('.', '_')], 0, 1)
            # get keys of each parameter in the layer (weight & bias)
            layer_keys = []
            for key in delta_dict_list[0].keys():
                if name in key:
                    layer_keys.append(key)

            for key in layer_keys:
                cross_delta = torch.stack([delta_dict_list[j][key].reshape(-1) for j in range(self.K)])
                for i in range(self.K):
                    self_delta = delta_dict_list[i][key].reshape(1, -1)
                    # cross_delta = torch.stack([delta_dict_list[j][key].reshape(-1) for j in range(self.K) if j != i])
                    cross_attn_delta = CrossAttention(self_delta, cross_delta, cross_delta)

                    gamma = layer_gamma[i]
                    ori_shape = delta_dict_list[i][key].shape
                    if self.norm:
                        new_delta = (1 - gamma) * delta_dict_list[i][key] + gamma * cross_attn_delta.reshape(ori_shape)
                    else:
                        if avg_delta_list is not None:
                            new_delta = avg_delta_list[i][key] + gamma * cross_attn_delta.reshape(ori_shape)
                        else:
                            new_delta = delta_dict_list[i][key] + gamma * cross_attn_delta.reshape(ori_shape)
                    new_param_dict_list[i][key] += new_delta

        return new_param_dict_list


def CrossAttention(q, k, v):
    scale = q.size(-1)**-0.5
    attn = (q @ k.transpose(-2, -1)) * scale
    attn = nn.Softmax(dim=-1)(attn)
    out = attn @ v

    return out


class HyperWeightALL(nn.Module):

    def __init__(self, K, init_gamma=1, norm=True):
        super(HyperWeightALL, self).__init__()
        self.K = K
        self.norm = norm

        # define parameters
        self.gamma = nn.Parameter(torch.ones(K) * init_gamma)

    def forward(self, flatten_last_param_list, flatten_delta, flatten_delta_update):
        flatten_new_param_list = copy.deepcopy(flatten_last_param_list)
        assert self.K == len(flatten_last_param_list)  # number of models

        gamma = torch.clamp(self.gamma, 0, 1)
        for i in range(self.K):
            if self.norm:
                flatten_new_param_list[i] += ((1 - gamma[i]) * flatten_delta[i] + gamma[i] * flatten_delta_update)
            else:
                flatten_new_param_list[i] += (flatten_delta[i] + gamma[i] * flatten_delta_update)

        return flatten_new_param_list
