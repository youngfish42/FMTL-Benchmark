from utils import get_output
from evaluation.save_img import save_img


class PerformanceMeter(object):
    """ A general performance meter which shows performance across one or more tasks """

    def __init__(self, dataname, tasks):
        self.tasks = tasks
        self.meters = {t: get_single_task_meter(dataname, t) for t in self.tasks}

    def reset(self):
        for t in self.tasks:
            self.meters[t].reset()

    def update(self, pred, gt):
        for t in self.tasks:
            self.meters[t].update(pred[t], gt[t])

    def get_score(self):
        eval_dict = {}
        for t in self.tasks:
            eval_dict[t] = self.meters[t].get_score()

        return eval_dict


def calculate_multi_task_performance(eval_dict, single_task_dict):
    assert (set(eval_dict.keys()) == set(single_task_dict.keys()))
    tasks = eval_dict.keys()
    num_tasks = len(tasks)
    mtl_performance = 0.0

    for task in tasks:
        mtl = eval_dict[task]
        stl = single_task_dict[task]

        if task == 'depth':  # rmse lower is better
            mtl_performance -= (mtl['rmse'] - stl['rmse']) / stl['rmse']

        elif task in ['semseg', 'sal', 'human_parts']:  # mIoU higher is better
            mtl_performance += (mtl['mIoU'] - stl['mIoU']) / stl['mIoU']

        elif task == 'normals':  # mean error lower is better
            mtl_performance -= (mtl['mean'] - stl['mean']) / stl['mean']

        elif task == 'edge':  # odsF higher is better
            mtl_performance += (mtl['odsF'] - stl['odsF']) / stl['odsF']

        else:
            raise NotImplementedError

    return mtl_performance / num_tasks


def get_single_task_meter(dataname, task):
    """ Retrieve a meter to measure the single-task performance """
    if task == 'semseg':
        from evaluation.eval_semseg import SemsegMeter
        return SemsegMeter(dataname)

    elif task == 'human_parts':
        from evaluation.eval_human_parts import HumanPartsMeter
        return HumanPartsMeter(dataname)

    elif task == 'normals':
        from evaluation.eval_normals import NormalsMeter
        return NormalsMeter()

    elif task == 'sal':
        from evaluation.eval_sal import SaliencyMeter
        return SaliencyMeter()

    elif task == 'depth':
        from evaluation.eval_depth import DepthMeter
        return DepthMeter(dataname)

    elif task == 'edge':  # Single task performance meter uses the loss (True evaluation is based on seism evaluation)
        from evaluation.eval_edge import EdgeMeter
        return EdgeMeter(pos_weight=0.95)

    else:
        raise NotImplementedError


def predict(dataname, meta, outputs, task, pred_dir, idx):
    output_task = get_output(outputs[task], task)
    preds = []
    for i in range(output_task.size(0)):
        # Cut image borders (padding area)
        pred = output_task[i]  # H, W or H, W, C
        ori_dim = (int(meta['size'][i][0]), int(meta['size'][i][1]))
        curr_dim = tuple(pred.shape[:2])

        if ori_dim != curr_dim:
            # Height and width of border
            delta_h = max(curr_dim[0] - ori_dim[0], 0)
            delta_w = max(curr_dim[1] - ori_dim[1], 0)

            # Location of original image
            loc_h = [delta_h // 2, (delta_h // 2) + ori_dim[0]]
            loc_w = [delta_w // 2, (delta_w // 2) + ori_dim[1]]

            pred = pred[loc_h[0]:loc_h[1], loc_w[0]:loc_w[1]]

        pred = pred.cpu().numpy()
        preds.append(pred)

    save_img(dataname, meta['file_name'], preds, task, pred_dir, str(idx))
