# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Sen Yang (yangsenius@seu.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil

import numpy as np
import random

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from core.function import train
from core.function import validate
from dataset.collater import collater
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
from utils.utils import get_model_summary, get_rank

import dataset
import models

import datetime
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')
    # parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--seed", type=int, default=22)
    args = parser.parse_args()

    return args


def init_distributed_mode(args):
    args.rank = int(os.environ["RANK"])
    args.world_size = int(os.environ['WORLD_SIZE'])
    args.local_rank = int(os.environ['LOCAL_RANK'])

    torch.cuda.set_device(args.local_rank)
    print('| distributed init (rank {}), gpu {}'.format(args.rank, args.local_rank), flush=True)
    torch.distributed.init_process_group(backend="nccl", init_method="env://", timeout=datetime.timedelta(minutes=60*12))


def cleanup():
    dist.destroy_process_group()


def main():
    args = parse_args()
    update_config(cfg, args)

    init_distributed_mode(args)
    local_rank = args.local_rank
    global_rank = args.rank
    num_tasks = args.world_size

    writer_dict = None
    logger, final_output_dir, tb_log_dir = create_logger(cfg, args.cfg, global_rank, 'train')
    if global_rank == 0:
        logger.info(pprint.pformat(args))
        logger.info(cfg)

        writer_dict = {
            'writer': SummaryWriter(log_dir=tb_log_dir),
            'train_global_steps': 0,
            'valid_global_steps': 0,
        }

        # copy model file
        this_dir = os.path.dirname(__file__)
        shutil.copy2(os.path.join(this_dir, '../lib/models', cfg.MODEL.NAME + '.py'), final_output_dir)
        
    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=True
    )

    model = model.cuda()
    optimizer = get_optimizer(cfg, model)
    # scaler = torch.cuda.amp.GradScaler()

    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(
        final_output_dir, 'checkpoint.pth'
    )

    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        last_epoch = checkpoint['epoch']

        if global_rank == 0:
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_file, checkpoint['epoch']))
            writer_dict['train_global_steps'] = checkpoint['train_global_steps']
            writer_dict['valid_global_steps'] = checkpoint['valid_global_steps']

        # model.load_state_dict(checkpoint['state_dict'])
        model.load_state_dict(checkpoint['best_state_dict'], strict=True)   #  strict=False FOR UNSeen Resolutions
        # model.load_state_dict({k.replace('module.',''):v for k, v in torch.load(checkpoint_file).items()}, strict=False, map_location=torch.device('cpu'))
        optimizer.load_state_dict(checkpoint['optimizer'])
        # scaler.load_state_dict(checkpoint['scaler'])

    # find_unused_parameters = False if cfg.MODEL.SINGLEFORMER_FIX else True
    ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, True,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    sampler_train = torch.utils.data.DistributedSampler(
        train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=cfg.TRAIN.SHUFFLE
    )
    collate_fn_train = collater(cfg.DATASET.MAX_PATCH, cfg.DATASET.PATCH_MODE)
    collate_fn_valid = collater(0)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, sampler=sampler_train,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU,
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        collate_fn=collate_fn_train
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU,
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        collate_fn=collate_fn_valid
    )

    best_perf = 0.0
    best_model = False
    last_epoch = -1

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, cfg.TRAIN.END_EPOCH, eta_min=cfg.TRAIN.LR_END, last_epoch=last_epoch)

    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        train_loader.sampler.set_epoch(epoch)
        if global_rank == 0:
            logger.info("=> current learning rate is {:.6f}".format(lr_scheduler.get_last_lr()[0]))
        # train for one epoch
        train(cfg, train_loader, ddp_model, criterion, optimizer, epoch, global_rank,
              final_output_dir, writer_dict)

        # evaluate on validation set
        if global_rank == 0:
            perf_indicator = validate(
                cfg, valid_loader, valid_dataset, ddp_model, criterion,
                final_output_dir, global_rank, writer_dict)
        
        lr_scheduler.step()

        if global_rank ==0 and perf_indicator >= best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False

        if global_rank == 0:
            logger.info('=> saving checkpoint to {}'.format(final_output_dir))
            save_checkpoint({
                'epoch': epoch + 1,
                'model': cfg.MODEL.NAME,
                'state_dict': ddp_model.state_dict(),
                'best_state_dict': ddp_model.module.state_dict(),
                'perf': perf_indicator,
                'optimizer': optimizer.state_dict(),
                # 'scaler': scaler.state_dict(),
                'train_global_steps': writer_dict['train_global_steps'],
                'valid_global_steps': writer_dict['valid_global_steps'],
            }, best_model, final_output_dir)

    if global_rank == 0:
        final_model_state_file = os.path.join(
            final_output_dir, 'final_state.pth'
        )
        logger.info('=> saving final model state to {}'.format(
            final_model_state_file)
        )
        torch.save(ddp_model.module.state_dict(), final_model_state_file)
        writer_dict['writer'].close()
    cleanup()
    print("#####\nTraining Done!\n#####")


if __name__ == '__main__':
    main()