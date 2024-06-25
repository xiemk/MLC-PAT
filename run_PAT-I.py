import argparse
import math
import os, sys
import random
import time
import json
import numpy as np
import datetime

import torch
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
import torch.optim
import torch.utils.data
from torch.cuda.amp import GradScaler, autocast


from torch.utils.tensorboard import SummaryWriter

from src_files.utils.logger import setup_logger
from src_files.utils.meter import AverageMeter, AverageMeterHMS, ProgressMeter
from src_files.utils.helper import function_mAP, add_weight_decay, get_raw_dict, ModelEma, clean_state_dict
from src_files.models.factory import create_model
from src_files.data.data import get_datasets


from torch.cuda.amp import GradScaler, autocast

NUM_CLASS = {'voc2007': 20, 'coco': 80, 'vg256': 256}

def get_args():
    parser = argparse.ArgumentParser(description='Clean ASL Training')

    # data
    parser.add_argument('--data_name', help='dataset name', default='coco', choices=['voc2007', 'coco','vg256'])
    parser.add_argument('--data_dir', help='dir of all datasets', default='/home/algroup/xmk/data')
    parser.add_argument('--image_size', default=448, type=int,
                        help='size of input images')
    parser.add_argument('--output', metavar='DIR', default='./outputs',
                        help='path to output folder')

    # model
    parser.add_argument('--model_name', default='tresnet_l')
    parser.add_argument('--pretrain_type', default='', type=str)
    parser.add_argument('--pretrain_dir', default='/home/algroup/xmk/PAT/pretrained', type=str)

    # train
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('-b', '--batch_size', default=64, type=int,
                        help='batch size')
    parser.add_argument('-p', '--print_freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 10)')

    # patching
    parser.add_argument('--n_grid', default=2, type=int)

    # random seed
    parser.add_argument('--seed', default=1, type=int,
                        help='seed for initializing training. ')
    
    parser.add_argument('--resume', help='dir of all datasets', default='/home/algroup/xmk/Patching/outputs/vg500/asl_dist_tresnet_l_21k_mldecoder_576_adam_0.0002_64_80/model_best.pth.tar')


    args = parser.parse_args()

    args.num_classes = NUM_CLASS[args.data_name]
    args.data_dir = os.path.join(args.data_dir, args.data_name) 
    
    args.output = os.path.join(args.output, args.data_name, f'PAT_I_{args.model_name}_{args.pretrain_type}_{args.image_size}_seed_{args.seed}')
    return args



def main():
    args = get_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

    os.makedirs(args.output, exist_ok=True)

    logger = setup_logger(output=args.output, color=False, name="XXX")
    logger.info("Command: "+' '.join(sys.argv))

    path = os.path.join(args.output, "config.json")
    with open(path, 'w') as f:
        json.dump(get_raw_dict(args), f, indent=2)
    logger.info("Full config saved to {}".format(path))

    return main_worker(args, logger)

def main_worker(args, logger):
    # build model
    print('creating model...')
    model = create_model(args).cuda()

    # Data loading
    train_dataset, val_dataset = get_datasets(args, patch=True)
    train_labels = train_dataset.Y
    pos_ratio = train_labels.sum(0)/train_labels.shape[0]
    print(pos_ratio)

    logger.info("len(train_dataset)): {}".format(len(train_dataset)))
    logger.info("len(val_dataset)): {}".format(len(val_dataset)))

    # Pytorch Data loader
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)
    
    # Set optimizer
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            state_dict = clean_state_dict(checkpoint['state_dict_ema'])
            logger.info(f"mAP: {checkpoint['best_mAP']}")
            model.load_state_dict(state_dict, strict=True)
            del checkpoint
            del state_dict
            torch.cuda.empty_cache()
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    
    mAP, APs, labels, probs = patch_validate(val_loader, model, args, logger)

    np.save(os.path.join(args.output, 'probs.npy'), probs)
    np.save(os.path.join(args.output, 'labels.npy'), labels)

    # preds = label_decision(labels, probs, pos_ratio)

    thresholding(labels, probs, logger)

    return 0

    patch_validate(val_loader, model, args, logger)

def calculate_metric(preds, labels):

    n_correct_pos = (labels*preds).sum(0)
    n_pred_pos = ((preds==1)).sum(0)
    n_true_pos = labels.sum(0)
    OP = n_correct_pos.sum()/n_pred_pos.sum()
    CP = np.nanmean(n_correct_pos/n_pred_pos)
    OR = n_correct_pos.sum()/n_true_pos.sum()
    CR = np.nanmean(n_correct_pos/n_true_pos)

    CF1 = (2 * CP * CR) / (CP + CR)
    OF1 = (2 * OP * OR) / (OP + OR)

    return CP, CR, CF1, OP, OR, OF1

def thresholding(labels, probs, logger):

    for thre in [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85, 0.9, 0.95]:
        preds = (probs>thre).astype(np.float32)
        metrics = calculate_metric(preds, labels)
        logger.info(f'{thre}, {np.round(np.array(metrics)*100, decimals=1)}')

    return 0

def label_decision(labels, probs, pos_ratio):
    
    sorted_probs = -np.sort(-probs, axis=0)
    indices = [int(x)-1 for x in pos_ratio*labels.shape[0]]
    thre_vec = sorted_probs[indices, range(probs.shape[1])]

    preds = (probs>=thre_vec).astype(np.float32)

    return preds


def weighted_sum(batch_size, logits_pat_1, logits_pat_2):
    split_list1 = torch.split(logits_pat_1, batch_size)          # [64,80] -> 4 * [16,80]
    logits_joint1 = torch.stack(split_list1, dim=1)                  # 4 * [16,80] -> [16, 4, 80]
    logits_sfmx1 = torch.softmax(logits_joint1, dim=1)               # [16, {4}, 80]
    
    split_list2 = torch.split(logits_pat_2, batch_size)          # [64,80] -> 4 * [16,80]
    logits_joint2 = torch.stack(split_list2, dim=1)                  # 4 * [16,80] -> [16, 4, 80]
    
    logits_joint = (logits_sfmx1 * logits_joint2).sum(dim=1)          # [16, 4, 80] -> [16,80]
    return logits_joint


@torch.no_grad()
def patch_validate(val_loader, model, args, logger):
    batch_time = AverageMeter('Time', ':5.3f')
    mem = AverageMeter('Mem', ':.0f', val_only=True)

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, mem],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    preds = []
    labels = []
        
    end = time.time()
    for i, (inputs, targets) in enumerate(val_loader):
        
        batch_size = targets.shape[0]
        inputs = torch.cat(inputs, dim=0).cuda(non_blocking=True)
    
        # compute output
        with autocast():
            outputs = model(inputs)
        
        outputs_ori = outputs[:batch_size]

        outputs_pat_1, outputs_pat_2 = outputs[batch_size:], outputs[batch_size:]

        outputs_pat = weighted_sum(batch_size, outputs_pat_1, outputs_pat_2)

        outputs = torch.sigmoid((outputs_ori+ outputs_pat)/2)

        # add list
        preds.append(outputs.detach().cpu())
        labels.append(targets.detach().cpu())

        # record memory
        mem.update(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % (2*args.print_freq) == 0:
            progress.display(i, logger)


    labels = torch.cat(labels).numpy()
    preds = torch.cat(preds).numpy()
    # calculate mAP
    mAP, APs= function_mAP(labels, preds)
    
    print("Calculating mAP:")  
    logger.info("  mAP: {:.2f}".format(mAP))

    return mAP, APs, labels, preds


if __name__ == '__main__':
    main()