import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from PIL import Image
from skimage.morphology import thin

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.save_img import (NYUDSegmentationMaskDecoder, VOCSegmentationMaskDecoder)

HUMAN_PART = {
    'hair': 1,
    'head': 1,
    'lear': 1,
    'lebrow': 1,
    'leye': 1,
    'lfoot': 6,
    'lhand': 4,
    'llarm': 4,
    'llleg': 6,
    'luarm': 3,
    'luleg': 5,
    'mouth': 1,
    'neck': 2,
    'nose': 1,
    'rear': 1,
    'rebrow': 1,
    'reye': 1,
    'rfoot': 6,
    'rhand': 4,
    'rlarm': 4,
    'rlleg': 6,
    'ruarm': 3,
    'ruleg': 5,
    'torso': 2
}
human_parts_category = 15
cat_part_file = json.load(open('/data/datasets/PASCALContext/json/pascal_part.json', 'r'))
cat_part_file[str(human_parts_category)] = HUMAN_PART


def get_parts_info(root, file_names):
    part_obj_dict = json.load(open(os.path.join(root, 'ImageSets/Parts/trainval.txt'), 'r'))
    has_human_parts = []
    for f_name in file_names:
        # part_obj_dict implies parts object category contained in image labels
        if human_parts_category in part_obj_dict[f_name]:
            has_human_parts.append(1)
        else:
            has_human_parts.append(0)

    return has_human_parts


def vis_convert(dataset, task, root, out_folder, file_name):
    if not os.path.exists(os.path.join(root, out_folder)):
        os.mkdir(os.path.join(root, out_folder))

    if dataset == "PASCALContext":
        if task == "edge":
            _tmp = sio.loadmat(os.path.join(root, 'pascal-context', 'trainval', file_name))
            _edge = cv2.Laplacian(_tmp['LabelMap'], cv2.CV_64F)
            _edge = thin(np.abs(_edge) > 0).astype(np.float32)
            cv2.imwrite(os.path.join(root, out_folder, file_name + '.png'), _edge * 255)
        elif task == "human_parts":
            _part_mat = sio.loadmat(os.path.join(root, 'human_parts', file_name))['anno'][0][0][1][0]
            _target = _inst_mask = None

            for _obj_ii in range(len(_part_mat)):
                has_human = _part_mat[_obj_ii][1][0][0] == human_parts_category
                has_parts = len(_part_mat[_obj_ii][3]) != 0

                if has_human and has_parts:
                    if _inst_mask is None:  # The first human in the image
                        _inst_mask = _part_mat[_obj_ii][2].astype(np.float32)  # Mask of positions contains human
                        _target = np.zeros(_inst_mask.shape)
                    else:
                        # If _inst_mask is not None, means there are more than one human in the image
                        # Take union of the humans
                        _inst_mask = np.maximum(_inst_mask, _part_mat[_obj_ii][2].astype(np.float32))

                    n_parts = len(_part_mat[_obj_ii][3][0])  # Number of parts object
                    for part_i in range(n_parts):
                        cat_part = str(_part_mat[_obj_ii][3][0][part_i][0][0])  # Name of part
                        mask_id = cat_part_file['15'][cat_part]
                        mask = _part_mat[_obj_ii][3][0][part_i][1].astype(bool)  # Position of part
                        _target[mask] = mask_id  # Label of part set as mask_id

            decoder = VOCSegmentationMaskDecoder(7)
            image = Image.fromarray(decoder(_target), mode='RGB')
            image.save(os.path.join(root, out_folder, file_name + '.png'))
        else:
            raise RuntimeError("Task error!")
    elif dataset == "NYUDv2":
        if task == "semseg":
            _semseg = np.array(Image.open(os.path.join(root, 'segmentation', file_name + '.png'))).astype(np.float32)
            decoder = NYUDSegmentationMaskDecoder(41)
            ss_img = Image.fromarray(decoder(_semseg), mode='RGB')
            ss_img.save(os.path.join(root, out_folder, file_name + ".png"))
        elif task == "depth":
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            _depth = np.load(os.path.join(root, 'depth', file_name + '.npy')).astype(np.float32)
            plt.imshow(_depth)
            plt.axis('off')
            plt.savefig(os.path.join(root, out_folder, file_name + ".png"))
        else:
            raise RuntimeError("Task error!")
    else:
        raise RuntimeError("Dataset error!")


if __name__ == "__main__":
    # dataset = "PASCALContext"
    dataset = "NYUDv2"
    root = os.path.join("/data/datasets", dataset)

    if dataset == "PASCALContext":
        with open(os.path.join(root, "ImageSets/Context/val.txt"), "r") as f:
            file_names = [x.strip() for x in f.readlines()]
        print(len(file_names))

        # parts info
        has_human_parts = get_parts_info(root, file_names)

        with ProcessPoolExecutor(max_workers=20) as executor:
            for i, file_name in enumerate(file_names):
                if has_human_parts[i]:
                    executor.submit(vis_convert, dataset, "human_parts", root, "parts_vis", file_name)
                executor.submit(vis_convert, dataset, "edge", root, "edge_vis", file_name)

        print("human_parts", len(os.listdir(os.path.join(root, "parts_vis"))))
        print("edge", len(os.listdir(os.path.join(root, "edge_vis"))))

    elif dataset == "NYUDv2":
        with open(os.path.join(root, "gt_sets/val.txt"), "r") as f:
            file_names = [x.strip() for x in f.readlines()]
        print(len(file_names))

        with ProcessPoolExecutor(max_workers=20) as executor:
            for file_name in file_names:
                executor.submit(vis_convert, dataset, "semseg", root, "semseg_vis", file_name)
                executor.submit(vis_convert, dataset, "depth", root, "depth_vis", file_name)

        print("semseg", len(os.listdir(os.path.join(root, "semseg_vis"))))
        print("depth", len(os.listdir(os.path.join(root, "depth_vis"))))
