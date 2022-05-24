# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import matplotlib.pyplot as plt
import matplotlib
import os
import math

import numpy as np
import torch
import torchvision
import cv2

from core.inference import get_max_preds


def save_batch_image_with_joints(batch_image, batch_joints, batch_joints_vis,
                                 file_name, nrow=8, padding=2):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            joints = batch_joints[k]
            joints_vis = batch_joints_vis[k]

            for joint, joint_vis in zip(joints, joints_vis):
                joint[0] = x * width + padding + joint[0]
                joint[1] = y * height + padding + joint[1]
                if joint_vis[0]:
                    cv2.circle(ndarr, (int(joint[0]), int(
                        joint[1])), 2, [255, 0, 0], 2)
            k = k + 1
    cv2.imwrite(file_name, ndarr)


def save_ori_image_with_joints(image, gruth, pred, batch_joints_vis,
                                 file_name, nrow=1, padding=2):
    '''
    batch_image: [channel, height, width]
    batch_joints: [nums, num_joints, 3],
    batch_joints_vis: [nums, num_joints, 1],
    }
    '''
    grid = torchvision.utils.make_grid(image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

    nmaps = batch_joints_vis.shape[0]

    k = 0
    while k < nmaps:
        joints_g = gruth[k]
        joints_p = pred[k]
        joints_vis = batch_joints_vis[k]

        for joint, joint_vis in zip(joints_p, joints_vis):
            joint[0] = padding + joint[0]
            joint[1] = padding + joint[1]
            if joint_vis[0]:
                cv2.circle(ndarr, (int(joint[0]), int(
                    joint[1])), 2, [255, 0, 0], 2)          # 蓝色

        for joint, joint_vis in zip(joints_g, joints_vis):
            joint[0] = padding + joint[0]
            joint[1] = padding + joint[1]
            if joint_vis[0]:
                cv2.circle(ndarr, (int(joint[0]), int(      # 红色
                    joint[1])), 2, [0, 0, 255], 2)
        k = k + 1

            
    # cv2.imshow('result.jpg', ndarr)
    plt.imshow(ndarr)
    plt.show()
    # cv2.imwrite(file_name, ndarr)


def save_batch_heatmaps(batch_image, batch_heatmaps, file_name,
                        normalize=True):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_heatmaps: ['batch_size, num_joints, height, width]
    file_name: saved file name
    '''
    if normalize:
        batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())

        batch_image.add_(-min).div_(max - min + 1e-5)

    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(1)
    heatmap_height = batch_heatmaps.size(2)
    heatmap_width = batch_heatmaps.size(3)

    grid_image = np.zeros((batch_size*heatmap_height,
                           (num_joints+1)*heatmap_width,
                           3),
                          dtype=np.uint8)

    preds, maxvals = get_max_preds(batch_heatmaps.detach().cpu().numpy())

    for i in range(batch_size):
        image = batch_image[i].mul(255)\
                              .clamp(0, 255)\
                              .byte()\
                              .permute(1, 2, 0)\
                              .cpu().numpy()
        heatmaps = batch_heatmaps[i].mul(255)\
                                    .clamp(0, 255)\
                                    .byte()\
                                    .cpu().numpy()

        resized_image = cv2.resize(image,
                                   (int(heatmap_width), int(heatmap_height)))

        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)
        for j in range(num_joints):
            cv2.circle(resized_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)
            heatmap = heatmaps[j, :, :]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            masked_image = colored_heatmap*0.7 + resized_image*0.3
            cv2.circle(masked_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)

            width_begin = heatmap_width * (j+1)
            width_end = heatmap_width * (j+2)
            grid_image[height_begin:height_end, width_begin:width_end, :] = \
                masked_image
            # grid_image[height_begin:height_end, width_begin:width_end, :] = \
            #     colored_heatmap*0.7 + resized_image*0.3

        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image

    cv2.imwrite(file_name, grid_image)


def compare_result_on_ori(truth_ori_coord, pred_ori_coord, meta, length, output_dir, i):
    import torchvision.transforms as transforms
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    tran = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    truth_ori_coord = torch.from_numpy(truth_ori_coord)
    truth_ori_coord = truth_ori_coord.split(length, dim=0)

    pred_ori_coord = torch.from_numpy(pred_ori_coord)
    pred_ori_coord = pred_ori_coord.split(length, dim=0)
    vis = meta['joints_vis'].split(length, dim=0)
    
    for idx, num in enumerate(length):
        data_numpy = cv2.imread(
            meta['image'][idx], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )
        data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)
        data_numpy = tran(data_numpy)

        gruth = truth_ori_coord[idx].numpy()
        pred = pred_ori_coord[idx].numpy()

        prefix = '{}_{}.jpg'.format(os.path.join(output_dir, str(i)), str(idx+1))

        query_locations = [np.array([p+0.5 for p in pred[i]]) for i in range(pred.shape[0])]
        
        res = plot_poses(data_numpy, query_locations)
        plt.imshow(res)
        plt.show()
        save_ori_image_with_joints(data_numpy, gruth, pred, vis[idx], prefix)


def save_patch_image_with_joints(batch_image, pred, truth, batch_joints_vis,
                                 file_name, nrow=8, padding=2):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            joints_p = pred[k]
            joints_g = truth[k]
            joints_vis = batch_joints_vis[k]

            for joint, joint_vis in zip(joints_p, joints_vis):
                joint[0] = x * width + padding + joint[0]
                joint[1] = y * height + padding + joint[1]
                if joint_vis[0]:
                    cv2.circle(ndarr, (int(joint[0]), int(
                        joint[1])), 2, [255, 0, 0], 2)

            for joint, joint_vis in zip(joints_g, joints_vis):
                joint[0] = x * width + padding + joint[0]
                joint[1] = y * height + padding + joint[1]
                if joint_vis[0]:
                    cv2.circle(ndarr, (int(joint[0]), int(
                        joint[1])), 2, [0, 0, 255], 2)
            k = k + 1
    cv2.imwrite(file_name, ndarr)


def compare_result_on_patch(input, pred, meta, output_dir, i):

    prefix = '{}_{}.jpg'.format(os.path.join(output_dir, 'res'), str(i))

    save_patch_image_with_joints(input, pred,  meta['joints'], meta['joints_vis'], prefix)



def save_debug_images(config, input, meta, target, joints_pred, output,
                      prefix):
    if not config.DEBUG.DEBUG:
        return

    if config.DEBUG.SAVE_BATCH_IMAGES_GT:
        save_batch_image_with_joints(
            input, meta['joints'], meta['joints_vis'],
            '{}_gt.jpg'.format(prefix)
        )
    if config.DEBUG.SAVE_BATCH_IMAGES_PRED:
        save_batch_image_with_joints(
            input, joints_pred, meta['joints_vis'],
            '{}_pred.jpg'.format(prefix)
        )
    if config.DEBUG.SAVE_HEATMAPS_GT:
        save_batch_heatmaps(
            input, target, '{}_hm_gt.jpg'.format(prefix)
        )
    if config.DEBUG.SAVE_HEATMAPS_PRED:
        save_batch_heatmaps(
            input, output, '{}_hm_pred.jpg'.format(prefix)
        )


# coco_keypoints_id['nose'] = 0
# coco_keypoints_id['l_eye']= 1
# coco_keypoints_id['r_eye'] = 2
# coco_keypoints_id['l_ear'] = 3
# coco_keypoints_id['r_ear'] = 4
# coco_keypoints_id['l_shoulder'] = 5
# coco_keypoints_id['r_shoulder'] = 6
# coco_keypoints_id['l_elbow'] = 7
# coco_keypoints_id['r_elbow'] = 8
# coco_keypoints_id['l_wrist'] = 9
# coco_keypoints_id['r_wrist'] = 10
# coco_keypoints_id['l_hip'] =11
# coco_keypoints_id['r_hip'] = 12
# coco_keypoints_id['l_knee'] = 13
# coco_keypoints_id['r_knee'] = 14
# coco_keypoints_id['l_ankle'] = 15
# coco_keypoints_id['r_ankle'] = 16


class plt_config:
    def __init__(self, dataset_name):
        if dataset_name == 'coco':
            self.n_kpt = 17
            # edge , color
            self.EDGES = [([15, 13], [255, 0, 0]),  # l_ankle -> l_knee
                          ([13, 11], [155, 85, 0]),  # l_knee -> l_hip
                          ([11, 5],  [155, 85, 0]),  # l_hip -> l_shoulder
                          ([12, 14], [0, 0, 255]),  # r_hip -> r_knee
                          ([14, 16], [17, 25, 10]),  # r_knee -> r_ankle
                          ([12, 6],  [0, 0, 255]),  # r_hip  -> r_shoulder
                          ([3, 1],   [0, 255, 0]),  # l_ear -> l_eye
                          ([1, 2],   [0, 255, 5]),  # l_eye -> r_eye
                          ([1, 0],   [0, 255, 170]),  # l_eye -> nose
                          ([0, 2],   [0, 255, 25]),  # nose -> r_eye
                          ([2, 4],   [0, 17, 255]),  # r_eye -> r_ear
                          ([9, 7],   [0, 220, 0]),  # l_wrist -> l_elbow
                          ([7, 5],   [0, 220, 0]),  # l_elbow -> l_shoulder
                          ([5, 6],   [125, 125, 155]), # l_shoulder -> r_shoulder
                          ([6, 8],   [25, 0, 55]),  # r_shoulder -> r_elbow
                          ([8, 10], [25, 0, 255])]  # r_elbow -> r_wrist
        elif dataset_name == 'jta':
            self.n_kpt = 22
            self.EDGES = [
                (0, 1),  # head_top -> head_center
                (1, 2),  # head_center -> neck
                (2, 3),  # neck -> right_clavicle
                (3, 4),  # right_clavicle -> right_shoulder
                (4, 5),  # right_shoulder -> right_elbow
                (5, 6),  # right_elbow -> right_wrist
                (2, 7),  # neck -> left_clavicle
                (7, 8),  # left_clavicle -> left_shoulder
                (8, 9),  # left_shoulder -> left_elbow
                (9, 10),  # left_elbow -> left_wrist
                (2, 11),  # neck -> spine0
                (11, 12),  # spine0 -> spine1
                (12, 13),  # spine1 -> spine2
                (13, 14),  # spine2 -> spine3
                (14, 15),  # spine3 -> spine4
                (15, 16),  # spine4 -> right_hip
                (16, 17),  # right_hip -> right_knee
                (17, 18),  # right_knee -> right_ankle
                (15, 19),  # spine4 -> left_hip
                (19, 20),  # left_hip -> left_knee
                (20, 21)  # left_knee -> left_ankle
            ]
        else:
            raise ValueError(
                "{} dataset is not supported".format(dataset_name))


def plot_poses(img_path, preds, config=plt_config('coco'), save_path=None, dataset_name='coco'):

    img = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    skeletons = [np.array([p+0.5 for p in preds[i]]) for i in range(preds.shape[0])]
    
    cmap = matplotlib.cm.get_cmap("hsv")
    canvas = img.copy()
    n_kpt = config.n_kpt
    for i in range(n_kpt):
        rgba = np.array(cmap(1 - i / n_kpt - 1.0 / n_kpt * 2))
        rgba[0:3] *= 255
        for j in range(len(skeletons)):
            if len(skeletons[j][i]) > 2 and skeletons[j][i, 2] > 0:
                cv2.circle(
                    canvas,
                    tuple(skeletons[j][i, 0:2].astype("int32")),
                    3,
                    (255, 255, 255),
                    thickness=-1,
                )

    stickwidth = 2
    for i in range(len(config.EDGES)):
        for j in range(len(skeletons)):
            edge = config.EDGES[i][0]
            color = config.EDGES[i][1]
            if len(skeletons[j][edge[0]]) > 2:
                if skeletons[j][edge[0], 2] == 0 or skeletons[j][edge[1], 2] == 0:
                    continue

            cur_canvas = canvas.copy()
            X = [skeletons[j][edge[0], 1], skeletons[j][edge[1], 1]]
            Y = [skeletons[j][edge[0], 0], skeletons[j][edge[1], 0]]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly(
                (int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1
            )
            cv2.fillConvexPoly(cur_canvas, polygon, color)
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
            
    cv2.imwrite(save_path, canvas)
