import numpy as np
import torch


class HumanPartsMeter(object):

    def __init__(self, dataname):
        assert (dataname == 'pascalcontext')
        self.database = dataname
        self.n_parts = 6
        self.tp = [0] * (self.n_parts + 1)
        self.fp = [0] * (self.n_parts + 1)
        self.fn = [0] * (self.n_parts + 1)

    @torch.no_grad()
    def update(self, pred, gt):
        pred, gt = pred.squeeze(), gt.squeeze()
        valid = (gt != 255)

        for i_part in range(self.n_parts + 1):
            tmp_gt = (gt == i_part)
            tmp_pred = (pred == i_part)
            self.tp[i_part] += torch.sum(tmp_gt & tmp_pred & (valid)).item()
            self.fp[i_part] += torch.sum(~tmp_gt & tmp_pred & (valid)).item()
            self.fn[i_part] += torch.sum(tmp_gt & ~tmp_pred & (valid)).item()

    def reset(self):
        self.tp = [0] * (self.n_parts + 1)
        self.fp = [0] * (self.n_parts + 1)
        self.fn = [0] * (self.n_parts + 1)

    def get_score(self):
        jac = [0] * (self.n_parts + 1)
        for i_part in range(0, self.n_parts + 1):
            jac[i_part] = float(self.tp[i_part]) / max(float(self.tp[i_part] + self.fp[i_part] + self.fn[i_part]), 1e-8)

        eval_result = dict()
        eval_result['mIoU'] = np.mean(jac) * 100

        return eval_result