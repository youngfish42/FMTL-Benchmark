# By Hanrong Ye
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import os

import cv2
import imageio
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as t_transforms
from PIL import Image

from .utils.mypath import MyPath
from .utils.configs import TRAIN_SCALE, TEST_SCALE


def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir 
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot, filename) for looproot, _, filenames in os.walk(rootdir) for filename in filenames
        if filename.endswith(suffix)
    ]


def imresize(img, size, mode, resample):
    size = (size[1], size[0])  # width, height
    _img = Image.fromarray(img)  #, mode=mode)
    _img = _img.resize(size, resample)
    _img = np.array(_img)
    return _img


class CITYSCAPES(data.Dataset):

    def __init__(self,
                 root=MyPath.db_root_dir('Cityscapes'),
                 dataidxs=None,
                 train=True,
                 is_transform=True,
                 augmentations=None,
                 task_list=['semseg', 'depth'],
                 ignore_index=255):

        if train:
            split = ['train']
            self.img_size = TRAIN_SCALE['cityscapes']
        else:
            split = ['val']
            self.img_size = TEST_SCALE['cityscapes']

        self.split = split
        self.root = root
        self.split_text = '+'.join(split)
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.n_classes = 19
        self.dd_label_map_size = [512, 1024]
        self.dataidxs = dataidxs

        self.task_list = task_list
        self.files = {}

        self.files[self.split_text] = []
        for _split in self.split:
            self.images_base = os.path.join(self.root, 'leftImg8bit', _split)
            self.annotations_base = os.path.join(self.root, 'gtFine', _split)
            self.files[self.split_text] += recursive_glob(rootdir=self.images_base, suffix='.png')
            self.depth_base = os.path.join(self.root, 'disparity', _split)
            # self.camera_base = os.path.join(self.root, 'camera', _split)
            # self.det_base = os.path.join(self.root, 'gtBbox3d', _split)
        ori_img_no = len(self.files[self.split_text])

        self.__build_truncated_dataset__()

        print("Found %d %s images, %d are used" % (ori_img_no, self.split_text, len(self.files[self.split_text])))

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.class_names = ['road', 'sidewalk', 'building', 'wall', 'fence',\
                            'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain',\
                            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', \
                            'motorcycle', 'bicycle']

        self.ignore_index = ignore_index
        self.class_map = dict(zip(self.valid_classes, range(19)))

        self.ori_img_size = [1024, 2048]
        self.label_dw_ratio = self.img_size[0] / self.ori_img_size[0]  # hacking

        if len(self.files[self.split_text]) < 2:
            raise Exception("No files for split=[%s] found in %s" % (self.split_text, self.images_base))

        # image to tensor
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.img_transform = t_transforms.Compose([t_transforms.ToTensor(), t_transforms.Normalize(mean, std)])

    def __len__(self):
        return len(self.files[self.split_text])

    def __build_truncated_dataset__(self):
        if self.dataidxs is not None:
            self.files[self.split_text] = [self.files[self.split_text][idx] for idx in self.dataidxs]

    def __getitem__(self, index):

        img_path = self.files[self.split_text][index].rstrip()
        lbl_path = os.path.join(self.annotations_base,
                                img_path.split(os.sep)[-2],
                                os.path.basename(img_path)[:-15] + 'gtFine_labelIds.png')
        # instance_path = os.path.join(self.annotations_base,
        #                              img_path.split(os.sep)[-2],
        #                              os.path.basename(img_path)[:-15] + 'gtFine_instanceIds.png')
        depth_path = os.path.join(self.depth_base,
                                  img_path.split(os.sep)[-2],
                                  os.path.basename(img_path)[:-15] + 'disparity.png')
        # camera_path = os.path.join(self.camera_base,
        #                            img_path.split(os.sep)[-2],
        #                            os.path.basename(img_path)[:-15] + 'camera.json')
        # det_path = os.path.join(self.det_base,
        #                         img_path.split(os.sep)[-2],
        #                         os.path.basename(img_path)[:-15] + 'gtBbox3d.json')

        img = cv2.imread(img_path).astype(np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        sample = {'image': img}
        sample['meta'] = {
            'file_name': img_path.split('.')[0].split('/')[-1],
            'size': (img.shape[0], img.shape[1]),
            'dd_label_map_size': self.dd_label_map_size,
            'scale_factor': np.array([self.img_size[1] / img.shape[1], self.img_size[0] / img.shape[0]]),  # in xy order
        }

        lbl = imageio.imread(lbl_path)

        if 'semseg' in self.task_list:
            sample['semseg'] = self.encode_segmap(lbl)

        if 'depth' in self.task_list:
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)  # disparity

            depth[depth > 0] = (depth[depth > 0] - 1) / 256  # disparity values

            # make the invalid idx to -1
            depth[depth == 0] = -1

            # assign the disparity of sky to zero
            sky_mask = lbl == 10
            depth[sky_mask] = 0

            # if False:
            #     # The model directly regress the depth value instead of disparity. Based on the official implementation: https://github.com/mcordts/cityscapesScripts/issues/55
            #     camera = json.load(open(camera_path))
            #     depth[depth >
            #           0] = camera["extrinsic"]["baseline"] * camera["intrinsic"]["fx"] / depth[depth > 0]  # real depth
            sample['depth'] = depth

        # if 'insseg' in self.task_list:
        #     ins = imageio.imread(instance_path)

        # if '3ddet' in self.task_list:
        #     # get 2D/3D detection labels
        #     det_labels, K_matrix, bbox_camera_params = self.load_det(det_path)
        #     sample['bbox_camera_params'] = bbox_camera_params
        #     sample['det_labels'] = det_labels
        #     sample['det_label_number'] = len(det_labels)
        #     sample['meta']['K_matrix'] = K_matrix

        # if 'insseg' in self.task_list:
        #     sample['ins'] = ins

        # if self.augmentations is not None:
        #     sample = self.augmentations(sample)

        if self.is_transform:
            sample = self.transform(sample)

        return sample

    def transform(self, sample):
        img = sample['image']
        if 'semseg' in self.task_list:
            lbl = sample['semseg']
        if 'depth' in self.task_list:
            depth = sample['depth']

        # img_ori_shape = img.shape[:2]
        img = img.astype(np.uint8)

        if self.img_size != self.ori_img_size:
            img = imresize(img, (self.img_size[0], self.img_size[1]), 'RGB', Image.BILINEAR)

        if 'semseg' in self.task_list:
            classes = np.unique(lbl)
            lbl = lbl.astype(float)
            # if self.img_size != self.ori_img_size:
            if self.dd_label_map_size != self.ori_img_size:
                lbl = imresize(lbl, (int(self.dd_label_map_size[0]), int(self.dd_label_map_size[1])), 'F',
                               Image.NEAREST)  # TODO(ozan) /8 is quite hacky
            lbl = lbl.astype(int)

            if not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.n_classes):
                print('after det', classes, np.unique(lbl))
                raise ValueError("Segmentation map contained invalid class values")
            lbl = torch.from_numpy(lbl).long()
            sample['semseg'] = lbl

        if 'depth' in self.task_list:
            # if self.img_size != self.ori_img_size:
            if self.dd_label_map_size != self.ori_img_size:
                depth = imresize(depth, (int(self.dd_label_map_size[0]), int(self.dd_label_map_size[1])), 'F',
                                 Image.NEAREST)
                # depth = depth * self.label_dw_ratio
            depth = np.expand_dims(depth, axis=0)
            depth = torch.from_numpy(depth).float()
            sample['depth'] = depth

        img = self.img_transform(img)
        sample['image'] = img

        return sample

    def encode_segmap(self, mask):
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        old_mask = mask.copy()
        for _validc in self.valid_classes:
            mask[old_mask == _validc] = self.class_map[_validc]
        return mask
