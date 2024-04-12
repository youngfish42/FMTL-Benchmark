import numpy as np
import torch


def jaccard(gt, pred, valid_pixels=None):
    """
    Jaccard (IoU),|A∩B|/|A∪B|,
    :param gt: Ground Truth
    :param pred: Prediction
    :param valid_pixels: A matrix indicates valid or not, defaults to None
    :return: Jaccard
    """

    assert gt.shape == pred.shape

    if valid_pixels is None:
        valid_pixels = np.zeros_like(gt)
    assert valid_pixels.shape == gt.shape

    gt = gt.astype(bool)
    pred = pred.astype(bool)
    valid_pixels = valid_pixels.astype(bool)
    if np.isclose(np.sum(gt & valid_pixels), 0) and np.isclose(np.sum(pred & valid_pixels), 0):
        return 1
    else:
        return np.sum(((gt & pred) & valid_pixels)) / \
               np.sum(((gt | pred) & valid_pixels), dtype=np.float32)


class SaliencyMeter(object):

    def __init__(self, ignore_index=255, threshold_step=0.05, beta_squared=0.3):
        self.ignore_index = ignore_index
        self.beta_squared = beta_squared
        self.thresholds = torch.arange(threshold_step, 1, threshold_step)
        self.true_positives = torch.zeros(len(self.thresholds))
        self.predicted_positives = torch.zeros(len(self.thresholds))
        self.actual_positives = torch.zeros(len(self.thresholds))
        self.all_jacs = []

    @torch.no_grad()
    def update(self, preds, target):
        """
        Update state with predictions and targets.

        Args:
            preds: Predictions from model [B, H, W]
            target: Ground truth values
        """
        preds = preds.float() / 255.

        if target.shape[1] == 1:
            target = target.squeeze(1)

        assert preds.shape == target.shape

        # squash logits into probabilities
        preds = torch.sigmoid(preds)

        valid_mask = (target != self.ignore_index)
        jacs = np.zeros(len(self.thresholds))

        for idx, thresh in enumerate(self.thresholds):
            # threshold probablities
            f_preds = (preds >= thresh).long()
            f_target = target.long()

            jacs[idx] = jaccard(f_target.cpu().numpy(), f_preds.cpu().numpy(), valid_mask.cpu().numpy())

            f_preds = torch.masked_select(f_preds, valid_mask)
            f_target = torch.masked_select(f_target, valid_mask)

            self.true_positives[idx] += torch.sum(f_preds * f_target).cpu()
            self.predicted_positives[idx] += torch.sum(f_preds).cpu()
            self.actual_positives[idx] += torch.sum(f_target).cpu()
        self.all_jacs.append(jacs)

    def get_score(self):
        """
        Computes F-scores over state and returns the max.
        """
        precision = self.true_positives.float() / (self.predicted_positives + 1e-8)
        recall = self.true_positives.float() / (self.actual_positives + 1e-8)

        num = (1 + self.beta_squared) * precision * recall
        denom = self.beta_squared * precision + recall

        # For the rest we need to take care of instances where the denom can be 0
        # for some classes which will produce nans for that class
        fscore = num / (denom + 1e-8)
        fscore[fscore != fscore] = 0

        eval_result = {'maxF': fscore.max().item() * 100}

        mIoUs = np.mean(np.array(self.all_jacs), 0)
        eval_result['mIoU'] = np.max(mIoUs) * 100

        return eval_result
