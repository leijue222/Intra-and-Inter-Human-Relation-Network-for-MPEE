# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Hanbin Dai (daihanbin.ac@gmail.com) and Feng Zhang (zhangfengwcy@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import os

import numpy as np
import torch

from core.evaluate import accuracy
from core.inference import get_final_preds
from utils.transforms import flip_back
from utils.vis import save_debug_images, plot_poses, compare_result_on_ori, compare_result_on_patch
from utils.utils import get_valid_output
from torch.cuda.amp import autocast as autocast
from pylsy import pylsytable

logger = logging.getLogger(__name__)


def train(config, train_loader, model, criterion, optimizer, epoch, global_rank, output_dir=None, writer_dict=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    loss_weights = config.MODEL.LOSS_WEIGHTS

    end = time.time()
    for i, (input, pos_mask, target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        length = meta['length'].numpy().tolist()
        target = target.cuda(non_blocking=True)      # [sum(length), 17, h, w]
        target_weight = target_weight.cuda(non_blocking=True)

        # compute output
        # with autocast():
        outputs = model(input, pos_mask, length)     # [bs*N, 17, h, w]
        if isinstance(outputs, dict):
            output = outputs['multi']
            loss_single = criterion(outputs['single'], target, target_weight, length)
            loss_multi = criterion(outputs['multi'], target, target_weight, length)
            # TODO add weight
            loss = loss_weights[0]*loss_single + loss_weights[1]*loss_multi
        else:
            output = outputs
            loss = criterion(output, target, target_weight, length)

        # compute gradient and do update step
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0 and global_rank == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, target, pred*4, output,
                              prefix)


def validate(config, val_loader, val_dataset, model, criterion, output_dir, global_rank, writer_dict=None):
    if global_rank != 0:
        return
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    all_preds = np.zeros(
        (1, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((1, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, pos_mask, target, target_weight, meta) in enumerate(val_loader):
            if config.TEST.USE_GT_BBOX:
                length = meta['length'].numpy().tolist()
            else:
                input, pos_mask, target, target_weight = input[0], pos_mask[0], target[0], target_weight[0]
                length = [1]*input.shape[0]

            # compute output
            # with autocast():
            outputs = model(input, pos_mask, length)     # [bs*N, 17, h, w]

            if isinstance(outputs, dict):
                output = outputs['multi']
            else:
                output = outputs

            if config.TEST.FLIP_TEST:
                # this part is ugly, because pytorch has not supported negative index
                # input_flipped = model(input[:, :, :, ::-1])
                input_flipped = np.flip(input.cpu().numpy(), 3).copy()
                pos_mask_flipped = np.flip(pos_mask.cpu().numpy(), 3).copy()

                input_flipped = torch.from_numpy(input_flipped).cuda()
                pos_mask_flipped = torch.from_numpy(pos_mask_flipped).cuda()

                # with autocast():
                outputs_flipped = model(input_flipped, pos_mask_flipped, length)
                if isinstance(outputs, dict):
                    output_flipped = outputs_flipped['multi']
                else:
                    output_flipped = outputs_flipped

                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

                output = (output + output_flipped) * 0.5

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            loss = criterion(output, target, target_weight, length)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if config.TEST.USE_GT_BBOX:
                c = meta['center'].numpy()
                s = meta['scale'].numpy()
                score = meta['score'].numpy()
            else:
                c = meta['center'][0].numpy()
                s = meta['scale'][0].numpy()
                score = meta['score'][0].numpy()

            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)

            # 可视化图片
            # length = meta['length'].numpy().tolist()
            # vis_preds = torch.from_numpy(preds)
            # vis_preds = torch.split(vis_preds, length, dim=0)
            # for img_path, vis_pred in zip(meta['image'], vis_preds):
            #     save_path = os.path.join(config.OUTPUT_DIR, 'coco', 'vis_val/', img_path.split('/')[-1])
            #     plot_poses(img_path, vis_pred.numpy(), save_path=save_path)
            
            # # 结束可视化
            # exit(1)

            # pred_ori_coord = preds.copy()
            # truth_ori_coord, _ = get_final_preds(config, target.clone().cpu().numpy(), c, s)
            # compare_result_on_ori(truth_ori_coord, pred_ori_coord, meta, length, os.path.join(output_dir, 'compare', 'ori'), i)

            # compare_result_on_patch(input, pred*4, meta, os.path.join(output_dir, 'compare', 'patch'), i)  # 对比patch和原图的点画的对错

            extend_pred = np.zeros((num_images, config.MODEL.NUM_JOINTS, 3), dtype=np.float32)
            extend_boxes = np.zeros((num_images, 6), dtype=np.float32)
            all_preds = np.concatenate((all_preds, extend_pred), axis=0)
            all_boxes = np.concatenate((all_boxes, extend_boxes), axis=0)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            
            for num, img in zip(length, meta['image']):
                image_path.extend([img for _ in range(num)])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                save_debug_images(config, input, meta, target, pred*4, output,
                                  prefix)
        all_preds = all_preds[:idx,...]
        all_boxes = all_boxes[:idx,...]

        assert all_preds.shape[0] == all_boxes.shape[0] == len(image_path)

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator

def validate_main_target(config, val_loader, val_dataset, model, criterion, output_dir, global_rank, writer_dict=None):
    if global_rank != 0:
        return
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    all_preds = np.zeros(
        (1, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((1, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    
    def get_target_person(tensor, length):
        target_list = []
        res_list = torch.split(tensor, length, dim=0)
        for res in res_list:
            target_list.append(torch.split(res, 1, dim=0)[0])
        return torch.cat(target_list, dim=0)
    
    with torch.no_grad():
        end = time.time()
        for i, (input, pos_mask, target, target_weight, meta) in enumerate(val_loader):
            if config.TEST.USE_GT_BBOX:
                length = meta['length'].numpy().tolist()
            else:
                input, pos_mask, target, target_weight = input[0], pos_mask[0], target[0], target_weight[0]
                length = [1]*input.shape[0]

            # compute output
            # with autocast():
            outputs = model(input, pos_mask, length)     # [bs*N, 17, h, w]

            if isinstance(outputs, dict):
                output = outputs['multi']
            else:
                output = outputs
                
            output = get_target_person(output, length)
            
            if config.TEST.FLIP_TEST:
                # this part is ugly, because pytorch has not supported negative index
                # input_flipped = model(input[:, :, :, ::-1])
                input_flipped = np.flip(input.cpu().numpy(), 3).copy()
                pos_mask_flipped = np.flip(pos_mask.cpu().numpy(), 3).copy()

                input_flipped = torch.from_numpy(input_flipped).cuda()
                pos_mask_flipped = torch.from_numpy(pos_mask_flipped).cuda()

                # with autocast():
                outputs_flipped = model(input_flipped, pos_mask_flipped, length)
                if isinstance(outputs, dict):
                    output_flipped = outputs_flipped['multi']
                else:
                    output_flipped = outputs_flipped
                    
                output_flipped = get_target_person(output_flipped, length)

                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

                output = (output + output_flipped) * 0.5

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)
            
            target = get_target_person(target, length)
            target_weight = get_target_person(target_weight, length)

            loss = criterion(output, target, target_weight, [len(length)])

            # num_images = input.size(0)
            num_images = len(length)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if config.TEST.USE_GT_BBOX:
                c = meta['center'].numpy()
                s = meta['scale'].numpy()
                score = meta['score'].numpy()
            else:
                c = meta['center'].numpy()
                s = meta['scale'].numpy()
                score = meta['score'].numpy()

            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)

            extend_pred = np.zeros((num_images, config.MODEL.NUM_JOINTS, 3), dtype=np.float32)
            extend_boxes = np.zeros((num_images, 6), dtype=np.float32)
            all_preds = np.concatenate((all_preds, extend_pred), axis=0)
            all_boxes = np.concatenate((all_boxes, extend_boxes), axis=0)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                # save_debug_images(config, input, meta, target, pred*4, output,
                #                   prefix)
        all_preds = all_preds[:idx,...]
        all_boxes = all_boxes[:idx,...]

        assert all_preds.shape[0] == all_boxes.shape[0] == len(image_path)

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator

# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = list(name_value.keys())
    values = list(name_value.values())
    table = pylsytable(names)
    logger.info('\nArch: ' + full_arch_name)
    for name, value in zip(names, values):
        table.add_data(str(name), round(value, 3))
    logger.info(table)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
