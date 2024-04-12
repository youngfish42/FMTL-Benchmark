import torch
from losses import BalancedBCELoss


class EdgeMeter(object):
    def __init__(self, pos_weight, ignore_index=255):
        self.loss = 0
        self.n = 0
        self.loss_function = BalancedBCELoss(pos_weight=pos_weight, ignore_index=ignore_index)
        self.ignore_index = ignore_index

    @torch.no_grad()
    def update(self, pred, gt):
        gt = gt.squeeze()
        pred = pred.squeeze()
        valid_mask = (gt != self.ignore_index)
        pred = pred[valid_mask]
        gt = gt[valid_mask]

        pred = pred.float().squeeze() / 255.
        loss = self.loss_function(pred, gt).item()
        numel = gt.numel()
        self.n += numel
        self.loss += numel * loss

    def reset(self):
        self.loss = 0
        self.n = 0

    def get_score(self):
        eval_dict = {'loss': self.loss / self.n}

        return eval_dict
