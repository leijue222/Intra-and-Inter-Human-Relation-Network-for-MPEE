# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Hanbin Dai (daihanbin.ac@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import random
import matplotlib.pyplot as plt

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from utils.transforms import get_affine_transform
from utils.transforms import affine_transform
from utils.transforms import fliplr_joints


logger = logging.getLogger(__name__)


class JointsDataset(Dataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        self.num_joints = 0
        self.pixel_std = 200
        self.flip_pairs = []
        self.parent_ids = []

        self.is_train = is_train
        self.root = root
        self.image_set = image_set

        self.output_path = cfg.OUTPUT_DIR
        self.data_format = cfg.DATASET.DATA_FORMAT

        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP
        self.num_joints_half_body = cfg.DATASET.NUM_JOINTS_HALF_BODY
        self.prob_half_body = cfg.DATASET.PROB_HALF_BODY
        self.color_rgb = cfg.DATASET.COLOR_RGB

        self.target_type = cfg.MODEL.TARGET_TYPE
        self.image_size = np.array(cfg.MODEL.IMAGE_SIZE)
        self.heatmap_size = np.array(cfg.MODEL.HEATMAP_SIZE)
        self.sigma = cfg.MODEL.SIGMA
        self.use_different_joints_weight = cfg.LOSS.USE_DIFFERENT_JOINTS_WEIGHT
        self.joints_weight = 1

        self.transform = transform
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.db = []

    def _get_db(self):
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        raise NotImplementedError

    def half_body_transform(self, joints, joints_vis):
        upper_joints = []
        lower_joints = []
        for joint_id in range(self.num_joints):
            if joints_vis[joint_id][0] > 0:
                if joint_id in self.upper_body_ids:
                    upper_joints.append(joints[joint_id])
                else:
                    lower_joints.append(joints[joint_id])

        if np.random.randn() < 0.5 and len(upper_joints) > 2:
            selected_joints = upper_joints
        else:
            selected_joints = lower_joints \
                if len(lower_joints) > 2 else upper_joints

        if len(selected_joints) < 2:
            return None, None

        selected_joints = np.array(selected_joints, dtype=np.float32)
        center = selected_joints.mean(axis=0)[:2]

        left_top = np.amin(selected_joints, axis=0)
        right_bottom = np.amax(selected_joints, axis=0)

        w = right_bottom[0] - left_top[0] + 1
        h = right_bottom[1] - left_top[1] + 1

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio

        scale = np.array(
            [
                w * 1.0 / self.pixel_std,
                h * 1.0 / self.pixel_std
            ],
            dtype=np.float32
        )

        scale = scale * 1.5

        return center, scale

    def show_images(self, image_list, titles=None, num_cols=None, scale=3, normalize=False):
        """ 一个窗口中绘制多张图像:
        Args: 
            images: 可以为一张图像(不要放在列表中)，也可以为一个图像列表
            titles: 图像对应标题、
            num_cols: 每行最多显示多少张图像
            scale: 用于调整图窗大小
            normalize: 显示灰度图时是否进行灰度归一化
        """
        # 加了下面2行后可以显示中文标题
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        # 多张图片显示
        if not isinstance(scale, tuple):
            scale = (scale, scale)

        num_imgs = len(image_list)
        if num_cols is None:
            num_cols = int(np.ceil((np.sqrt(num_imgs))))
        num_rows = (num_imgs - 1) // num_cols + 1

        idx = list(range(num_imgs))
        _, figs = plt.subplots(num_rows, num_cols,
                            figsize=(scale[1] * num_cols, scale[0] * num_rows))
        for f, i, img in zip(figs.flat, idx, image_list):
            if len(img.shape) == 3:
                # opencv库中函数生成的图像为BGR通道，需要转换一下
                # B, G, R = cv2.split(img)
                # img = cv2.merge([R, G, B])
                f.imshow(img)
            elif len(img.shape) == 2:
                # pyplot显示灰度需要加一个参数
                if normalize:
                    f.imshow(img, cmap='gray')
                else:
                    f.imshow(img, cmap='gray', vmin=0, vmax=255)
            else:
                raise TypeError("Invalid shape " +
                                str(img.shape) + " of image data")
            if titles is not None:
                f.set_title(titles[i], y=-0.15)
            f.axes.get_xaxis().set_visible(True)
            f.axes.get_yaxis().set_visible(True)
        # 将不显示图像的fig移除，不然会显示多余的窗口
        if len(figs.shape) == 1:
            figs = figs.reshape(-1, figs.shape[0])
        for i in range(num_rows * num_cols - num_imgs):
            figs[num_rows - 1, num_imgs % num_cols + i].remove()
        plt.show()

    def get_position(self, shape, boxs, type='single', flipFlag=False):
        rectangle = np.zeros(shape, dtype="uint8")
        if type == 'single':
            x, y, w, h = boxs[:4]
            cv2.rectangle(rectangle, (int(x), int(y)), (int(x+w), int(y+h)), 255, -1)
        elif type == 'multi':
            for item in boxs:
                x, y, w, h = item['box'][:4]
                cv2.rectangle(rectangle, (int(x), int(y)), (int(x+w), int(y+h)), 255, -1)
        if flipFlag:
            rectangle = rectangle[:, ::-1]
            # print('!!position翻转')
        return rectangle

    def rotate_bound(self, image, angle):
        # grab the dimensions of the image and then determine the
        # center
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
    
        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
    
        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
    
        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
    
        # perform the actual rotation and return the image
        return cv2.warpAffine(image, M, (nW, nH))

    def __len__(self,):
        return len(self.db)

    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])

        image_file = db_rec['image']
        filename = db_rec['filename'] if 'filename' in db_rec else ''

        if self.data_format == 'zip':
            from utils import zipreader
            data_numpy = zipreader.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )
        else:
            data_numpy = cv2.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )

        if self.color_rgb:
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

        if data_numpy is None:
            logger.error('=> fail to read {}'.format(image_file))
            raise ValueError('Fail to read {}'.format(image_file))

        image_list = [data_numpy]
        image_title = ['ori']

        flipFlag = False
        r = 0
        if self.is_train:
            # 图像增强取得随机旋转的值
            rf = self.rotation_factor
            r = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
            if random.random() <= 0.6 else 0
            
            sf = self.scale_factor
            sf_ratio = np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
            half_trans_flag = np.random.rand() < self.prob_half_body

            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                flipFlag = True
                image_list.append(data_numpy)
                image_title.append('flip')

        input_list, multi_pos_list, pos_mask_list, target_list, target_weight_list = [], [], [], [], []
        meta = {
            'image': image_file,
            'filename': filename,
            'rotation': r,
            'imgnum': [],
            'joints': [],
            'joints_vis': [],
            'center': [],
            'scale': [],
            'score': [],
            'box': []
        }
        
        multi_mask_numpy = self.get_position(data_numpy.shape[0:2], db_rec['annos'], type='multi', flipFlag=flipFlag)

        image_list.append(multi_mask_numpy)
        image_title.append('multi_position')
        for anno in db_rec['annos']:

            imgnum = anno['imgnum'] if 'imgnum' in anno else ''
            joints = anno['joints_3d']
            joints_vis = anno['joints_3d_vis']

            c = anno['center']
            s = anno['scale']
            score = anno['score'] if 'score' in anno else 1
                
            if self.is_train:
                if flipFlag:    # 图片已经翻转，关节点也跟着翻转，同时改变中心点
                    joints, joints_vis = fliplr_joints(joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
                    c[0] = data_numpy.shape[1] - c[0] - 1

                s = s * sf_ratio

                if (np.sum(joints_vis[:, 0]) > self.num_joints_half_body
                    and half_trans_flag):
                    c_half_body, s_half_body = self.half_body_transform(
                        joints, joints_vis
                    )

                    if c_half_body is not None and s_half_body is not None:
                        c, s = c_half_body, s_half_body
                    
            joints_heatmap = joints.copy()
            trans = get_affine_transform(c, s, r, self.image_size)
            trans_heatmap = get_affine_transform(c, s, r, self.heatmap_size)

            input = cv2.warpAffine(
                data_numpy,
                trans,
                (int(self.image_size[0]), int(self.image_size[1])),
                flags=cv2.INTER_LINEAR)

            input_mask = cv2.warpAffine(
                multi_mask_numpy,
                trans,
                (int(self.image_size[0]), int(self.image_size[1])),
                flags=cv2.INTER_LINEAR)

            image_list.append(input)
            image_title.append('input ' + str(imgnum))

            image_list.append(input_mask)
            image_title.append('input_mask ' + str(imgnum))

            for i in range(self.num_joints):
                if joints_vis[i, 0] > 0.0:  # 跟着input一起trans
                    joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)
                    joints_heatmap[i, 0:2] = affine_transform(joints_heatmap[i, 0:2], trans_heatmap)

            # 生成input在图片中对应的position mask
            pos_mask_numpy = self.get_position(data_numpy.shape[0:2], anno['box'], type='single', flipFlag=flipFlag)
            pos_mask = self.rotate_bound(pos_mask_numpy, r)
            pos_mask = cv2.resize(pos_mask, self.image_size)
            image_list.append(pos_mask)
            image_title.append('pos_mask')    

            if self.transform:  # ToTensor and Normalize
                input = self.transform(input)
                pos_mask = self.mask_transform(pos_mask)
                input_mask = self.mask_transform(input_mask)

            target, target_weight = self.generate_target(joints_heatmap, joints_vis)

            target = torch.from_numpy(target)
            target_weight = torch.from_numpy(target_weight)

            meta['imgnum'].append(imgnum)
            meta['joints'].append(joints)
            meta['joints_vis'].append(joints_vis)
            meta['center'].append(c)
            meta['scale'].append(s)
            meta['score'].append(score)
            meta['box'].append(anno['box'])

            input_list.append(input)
            multi_pos_list.append(input_mask)
            pos_mask_list.append(pos_mask)
            target_list.append(target)
            target_weight_list.append(target_weight)

        assert len(db_rec['annos']) == len(input_list)
        
        # self.show_images(image_list, image_title)
        return input_list, pos_mask_list, target_list, target_weight_list, meta
        # return input_list, multi_pos_list, target_list, target_weight_list, meta


    def select_data(self, db):
        db_selected = []
        for rec in db:
            num_vis = 0
            joints_x = 0.0
            joints_y = 0.0
            for joint, joint_vis in zip(
                    rec['joints_3d'], rec['joints_3d_vis']):
                if joint_vis[0] <= 0:
                    continue
                num_vis += 1

                joints_x += joint[0]
                joints_y += joint[1]
            if num_vis == 0:
                continue

            joints_x, joints_y = joints_x / num_vis, joints_y / num_vis

            area = rec['scale'][0] * rec['scale'][1] * (self.pixel_std**2)
            joints_center = np.array([joints_x, joints_y])
            bbox_center = np.array(rec['center'])
            diff_norm2 = np.linalg.norm((joints_center-bbox_center), 2)
            ks = np.exp(-1.0*(diff_norm2**2) / ((0.2)**2*2.0*area))

            metric = (0.2 / 16) * num_vis + 0.45 - 0.2 / 16
            if ks > metric:
                db_selected.append(rec)

        logger.info('=> num db: {}'.format(len(db)))
        logger.info('=> num selected db: {}'.format(len(db_selected)))
        return db_selected


    def generate_target(self, joints, joints_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros((self.num_joints,
                               self.heatmap_size[1],
                               self.heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = self.sigma * 3

            for joint_id in range(self.num_joints):
                target_weight[joint_id] = \
                    self.adjust_target_weight(joints[joint_id], target_weight[joint_id], tmp_size)
                
                if target_weight[joint_id] == 0:
                    continue

                mu_x = joints[joint_id][0]
                mu_y = joints[joint_id][1]
                
                x = np.arange(0, self.heatmap_size[0], 1, np.float32)
                y = np.arange(0, self.heatmap_size[1], 1, np.float32)
                y = y[:, np.newaxis]

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id] = np.exp(- ((x - mu_x) ** 2 + (y - mu_y) ** 2) / (2 * self.sigma ** 2))

        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        return target, target_weight


    def adjust_target_weight(self, joint, target_weight, tmp_size):
        # feat_stride = self.image_size / self.heatmap_size
        mu_x = joint[0]
        mu_y = joint[1]
        # Check that any part of the gaussian is in-bounds
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                or br[0] < 0 or br[1] < 0:
            # If not, just return the image as is
            target_weight = 0

        return target_weight
