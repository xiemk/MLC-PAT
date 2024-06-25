import argparse
import math
import os, sys
import random
import time
import json
import numpy as np

from typing import List

import torch
from torch.optim import lr_scheduler
import torch.optim
import torch.utils.data
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist

from torch.utils.tensorboard import SummaryWriter

from src_files.utils.logger import setup_logger
from src_files.utils.meter import AverageMeter, AverageMeterHMS, ProgressMeter
from src_files.utils.helper import add_weight_decay, ModelEma, sl_mAP
from src_files.utils.losses import AsymmetricLoss
from src_files.models.factory import create_model
from src_files.data.data import get_datasets

from torch.cuda.amp import GradScaler, autocast

NUM_CLASS = {'voc2007': 20, 'coco': 80,  'vg256':256}

def get_args():
    parser = argparse.ArgumentParser(description='Clean ASL Training')

    # data
    parser.add_argument('--data_name', help='dataset name', default='coco', choices=['voc2007', 'coco', 'vg256'])
    parser.add_argument('--data_dir', help='dir of all datasets', default='/home/algroup/xmk/data')
    parser.add_argument('--image_size', default=448, type=int,
                        help='size of input images')
    parser.add_argument('--output', metavar='DIR', default='./outputs',
                        help='path to output folder')

    # model
    parser.add_argument('--model_name', default='tresnetl_v2')
    parser.add_argument('--backbone', default='tresnetl_v2')
    parser.add_argument('--pretrain_type', default='in21k', choices=['in1k', 'in21k','oi'])
    parser.add_argument('--pretrain_dir', default='/home/algroup/xmk/PAT/pretrained', type=str)
    parser.add_argument('--ema_decay', default=0.9997, type=float, metavar='M',
                        help='decay of model ema')

    # train
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--epochs', default=80, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=64, type=int,
                        help='batch size')
    parser.add_argument('--optim', default='adamw', type=str,
                        help='optimizer used')
    parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-2)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print_freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--early_stop', action='store_true', default=True,
                        help='apply early stop')
    parser.add_argument('--gamma_neg', default=4, type=int)
    parser.add_argument('--clip', default=0.05, type=float)

    # distribution training
    parser.add_argument('--distributed', action='store_true', help='using dataparallel')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')

    # Pathcing
    parser.add_argument('--n_grid', default=2, type=int)
    parser.add_argument('--logits_attention', default='cross', type=str)
    parser.add_argument('--temperature', default=1.0, type=float, help='temperature for softmax')

    # random seed
    parser.add_argument('--seed', default=1, type=int,
                        help='seed for initializing training. ')


    args = parser.parse_args()

    args.num_classes = NUM_CLASS[args.data_name]
    args.data_dir = os.path.join(args.data_dir, args.data_name) 
    
    args.output = os.path.join(args.output, args.data_name, f'rep_patching_dist_{args.logits_attention}_{args.model_name}_{args.backbone}_{args.pretrain_type}_{args.image_size}_{args.optim}_{args.lr}_{args.weight_decay}_{args.gamma_neg}_{args.clip}_{args.batch_size}_{args.epochs}_seed_{args.seed}')

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

    # setup dist training
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0  # global rank

    if args.distributed:
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        print('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.' % (args.rank, args.world_size))
    else:
        print('Training with a single process on 1 GPUs.')
    assert args.rank >= 0
    
    logger = setup_logger(output=args.output, distributed_rank=dist.get_rank(), color=False, name="Coco")
    logger.info("Command: "+' '.join(sys.argv))
    if dist.get_rank() == 0:
        path = os.path.join(args.output, "config.json")
        with open(path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        logger.info("Full config saved to {}".format(path))
        os.makedirs(os.path.join(args.output, 'tmpdata'), exist_ok=True)

    return main_worker(args, logger)

def main_worker(args, logger):
    # build model

    
    logger.info('creating model...')
    model = create_model(args).cuda()
    # logger.info("{}".format(args.pretrain_path))

    ema_m = ModelEma(model, args.ema_decay)  # 0.9997^641=0.82

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], broadcast_buffers=False)


    # Data loading code
    # COCO Data loading
    train_dataset, val_dataset = get_datasets(args, patch=True)
    logger.info("len(train_dataset)): {}".format(len(train_dataset)))
    logger.info("len(val_dataset)): {}".format(len(val_dataset)))
   
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    assert args.batch_size // dist.get_world_size() == args.batch_size / dist.get_world_size(), 'Batch size is not divisible by num of gpus.'

    # Pytorch Data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size // dist.get_world_size(), 
        shuffle=not args.distributed,
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size= 128 // dist.get_world_size(), 
        shuffle=False,
        num_workers=args.workers, pin_memory=False, sampler=val_sampler)
    
    epoch_time = AverageMeterHMS('TT')
    eta = AverageMeterHMS('ETA', val_only=True)
    mAPs = AverageMeter('mAP', ':5.5f', val_only=True)
    mAPs_ema = AverageMeter('mAP_ema', ':5.5f', val_only=True)
    progress = ProgressMeter(
        args.epochs,
        [eta, epoch_time, mAPs, mAPs_ema],
        prefix='=> Test Epoch: ')

    # Set optimizer
    optimizer = set_optimizer(model, args)
    args.steps_per_epoch = len(train_loader)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=args.steps_per_epoch, epochs=args.epochs, pct_start=0.2)

    # Set loss func
    criterion = AsymmetricLoss(gamma_neg=args.gamma_neg, gamma_pos=0, clip=args.clip, disable_torch_grad_focal_loss=True)
   
    end = time.time()
    best_epoch = -1
    best_regular_mAP = 0
    best_regular_epoch = -1
    best_ema_mAP = 0
    regular_mAP_list = []
    ema_mAP_list = []
    best_mAP = 0

    # tensorboard
    summary_writer = SummaryWriter(log_dir=args.output)

    for epoch in range(args.epochs):

        torch.cuda.empty_cache()

        # train for one epoch
        loss = train(train_loader, model, ema_m, optimizer, scheduler, epoch, args, logger, criterion)
    

        # tensorboard logger
        if summary_writer:
            summary_writer.add_scalar('train_loss', loss, epoch)
            summary_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # evaluate on validation set
        mAP, APs, mAP_ema, APs_ema = validate(val_loader, model, ema_m, args, logger)
        mAPs.update(mAP)
        mAPs_ema.update(mAP_ema)
        epoch_time.update(time.time() - end)
        end = time.time()
        eta.update(epoch_time.avg * (args.epochs - epoch - 1))

        regular_mAP_list.append(mAP)
        ema_mAP_list.append(mAP_ema)

        progress.display(epoch, logger)

        if summary_writer:
            # tensorboard logger
            summary_writer.add_scalar('val_mAP', mAP, epoch)
            summary_writer.add_scalar('val_mAP_ema', mAP_ema, epoch)

        # remember best (regular) mAP and corresponding epochs
        if mAP > best_regular_mAP:
            best_regular_mAP = max(best_regular_mAP, mAP)
            best_regular_epoch = epoch
        if mAP_ema > best_ema_mAP:
            best_ema_mAP = max(mAP_ema, best_ema_mAP)
            best_ema_epoch = epoch

        if mAP_ema > mAP:
            mAP = mAP_ema

        state_dict = model.state_dict()
        state_dict_ema = ema_m.module.state_dict()

        is_best = mAP > best_mAP
        if is_best:
            best_epoch = epoch
        best_mAP = max(mAP, best_mAP)

        logger.info("{} | Set best mAP {} in ep {}".format(epoch, best_mAP, best_epoch))
        logger.info("   | best regular mAP {} in ep {}".format(best_regular_mAP, best_regular_epoch))

        if dist.get_rank() == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': state_dict,
                'state_dict_ema': state_dict_ema,
                'best_mAP': best_mAP,
                'AP': APs,
                'AP_ema': APs_ema,
                'optimizer' : optimizer.state_dict(),
            }, is_best=is_best, filename=os.path.join(args.output, 'model_best.pth.tar'))

            if math.isnan(loss):
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': state_dict,
                    'state_dict_ema': state_dict_ema,
                    'best_mAP': best_mAP,
                    'AP': APs,
                    'AP_ema': APs_ema,
                    'optimizer' : optimizer.state_dict(),
                }, is_best=is_best, filename=os.path.join(args.output, 'model_best_nan.pth.tar'))
                logger.info('Loss is NaN, break')
                sys.exit(1)


        # early stop
        if args.early_stop:
            if best_epoch >= 0 and epoch - max(best_epoch, best_regular_epoch) > 1:
                if len(ema_mAP_list) > 1 and ema_mAP_list[-1] < best_ema_mAP:
                    logger.info("epoch - best_epoch = {}, stop!".format(epoch - best_epoch))
                    if dist.get_rank() == 0 and args.kill_stop:
                        filename = sys.argv[0].split(' ')[0].strip()
                        killedlist = kill_process(filename, os.getpid())
                        logger.info("Kill all process of {}: ".format(filename) + " ".join(killedlist)) 
                    break


    print("Best mAP:", best_mAP)

    if summary_writer:
        summary_writer.close()
    
    return 0

def kill_process(filename:str, holdpid:int) -> List[str]:
    import subprocess, signal
    res = subprocess.check_output("ps aux | grep {} | grep -v grep | awk '{{print $2}}'".format(filename), shell=True, cwd="./")
    res = res.decode('utf-8')
    idlist = [i.strip() for i in res.split('\n') if i != '']
    print("kill: {}".format(idlist))
    for idname in idlist:
        if idname != str(holdpid):
            os.kill(int(idname), signal.SIGKILL)
    return idlist

def set_optimizer(model, args):

    if args.optim == 'adam':
        parameters = add_weight_decay(model, args.weight_decay)
        optimizer = torch.optim.Adam(params=parameters, lr=args.lr, weight_decay=0)  # true wd, filter_bias_and_bn
    elif args.optim == 'adamw':
        param_dicts = [
            {"params": [p for n, p in model.named_parameters() if p.requires_grad]},
        ]
        optimizer = getattr(torch.optim, 'AdamW')(
            param_dicts,
            args.lr,
            betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay
        )

    return optimizer

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    if is_best:
        torch.save(state, filename)

def weighted_sum(batch_size, logits_pat_1, logits_pat_2):
    split_list1 = torch.split(logits_pat_1, batch_size)          # [64,80] -> 4 * [16,80]
    logits_joint1 = torch.stack(split_list1, dim=1)                  # 4 * [16,80] -> [16, 4, 80]
    logits_sfmx1 = torch.softmax(logits_joint1, dim=1)               # [16, {4}, 80]
    
    split_list2 = torch.split(logits_pat_2, batch_size)          # [64,80] -> 4 * [16,80]
    logits_joint2 = torch.stack(split_list2, dim=1)                  # 4 * [16,80] -> [16, 4, 80]
    
    logits_joint = (logits_sfmx1 * logits_joint2).sum(dim=1)          # [16, 4, 80] -> [16,80]
    return logits_joint


def train(train_loader, model, ema_m, optimizer, scheduler, epoch, args, logger, criterion):
    scaler = GradScaler()
    
    losses = AverageMeter('Loss', ':5.3f')
    # lr = AverageMeter('LR', ':.3e', val_only=True)
    # mem = AverageMeter('Mem', ':.0f', val_only=True)
    # progress = ProgressMeter(
    #     args.steps_per_epoch,
    #     [lr, losses, mem],
    #     prefix="Epoch: [{}/{}]".format(epoch, args.epochs))

    def get_learning_rate(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    # lr.update(get_learning_rate(optimizer))
    # logger.info("lr:{}".format(get_learning_rate(optimizer)))

    # switch to train mode
    model.train()

    for i, (inputs, targets) in enumerate(train_loader):

        # **********************************************compute loss*************************************************************
        batch_size = targets.shape[0]

        inputs = torch.cat(inputs, dim=0).cuda(non_blocking=True)
        # print(inputs.shape)
        # inputs_ori = inputs[0].cuda(non_blocking=True)
        # inputs_pat = torch.cat(inputs[1:], dim=0).cuda(non_blocking=True)

        # print(inputs_pat.size())
        
        targets = targets.cuda()
        # mixed precision ---- compute outputs
        with autocast():
            logits = model(inputs)
        
        
        logits_ori = logits[0][:batch_size]
        
        if args.logits_attention == 'self':
            logits_pat_1, logits_pat_2 = logits[1][batch_size:], logits[1][batch_size:]
        elif args.logits_attention == 'cross':
            logits_pat_1, logits_pat_2 = logits[1][batch_size:], logits[2][batch_size:]
        else:
            print("attention: {} not found !!".format(args.logits_attention))
            exit(-1)


        logits_pat = weighted_sum(batch_size, logits_pat_1, logits_pat_2)

        # if args.logitsum:
        #     logits = (logits_ori+logits_pat)/2
        #     loss = criterion(logits, targets)
        # else:
        # loss_ori = criterion(logits_ori, targets)
        # loss_pat = criterion(logits_pat, targets)
        loss = criterion(logits_ori, targets) + criterion(logits_pat, targets)

        # record loss
        losses.update(loss.item(), batch_size)
        # mem.update(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)

        # compute gradient and do SGD step
        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # one cycle learning rate
        scheduler.step()
        # lr.update(get_learning_rate(optimizer))
        ema_m.update(model)

        
        if i % args.print_freq == 0:
            logger.info('Epoch [{}/{}], Step [{}/{}], LR {:.3e}, Loss: {:.2f}'
                    .format(epoch, args.epochs, str(i).zfill(3), str(args.steps_per_epoch).zfill(3),
                            scheduler.get_last_lr()[0], loss.item()))

    return losses.avg


@torch.no_grad()
def validate(val_loader, model, ema_m, args, logger):
    batch_time = AverageMeter('Time', ':5.3f')
    mem = AverageMeter('Mem', ':.0f', val_only=True)

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, mem],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    preds = []
    preds_ema = []
    labels = []
        
    end = time.time()
    for i, (inputs, targets) in enumerate(val_loader):

        batch_size = targets.shape[0]
        inputs = torch.cat(inputs, dim=0).cuda(non_blocking=True)
    
        # compute output
        with autocast():
            outputs = model(inputs)
            outputs_ema = ema_m.module(inputs)
   
        outputs_ori = outputs[0][:batch_size]

        if args.logits_attention == 'self':
            outputs_pat_1, outputs_pat_2 = outputs[1][batch_size:], outputs[1][batch_size:]
            outputs_ema_pat_1, outputs_ema_pat_2 = outputs_ema[1][batch_size:], outputs_ema[1][batch_size:]
        elif args.logits_attention == 'cross':
            outputs_pat_1, outputs_pat_2 = outputs[1][batch_size:], outputs[2][batch_size:]
            outputs_ema_pat_1, outputs_ema_pat_2 = outputs_ema[1][batch_size:], outputs_ema[2][batch_size:]
        else:
            print("attention: {} not found !!".format(args.logits_attention))
            exit(-1)
            
        # outputs_pat_1, outputs_pat_2 = outputs[1][batch_size:], outputs[2][batch_size:]
        outputs_pat = weighted_sum(batch_size, outputs_pat_1, outputs_pat_2)
        outputs = torch.sigmoid((outputs_ori+ outputs_pat)/2)

        outputs_ema_ori = outputs_ema[0][:batch_size]
        # outputs_ema_pat_1, outputs_ema_pat_2 = outputs_ema[1][batch_size:], outputs_ema[2][batch_size:]
        outputs_ema_pat = weighted_sum(batch_size, outputs_ema_pat_1, outputs_ema_pat_2)
        outputs_ema = torch.sigmoid((outputs_ema_ori+ outputs_ema_pat)/2)


        # add list
        preds.append(outputs.detach().cpu())
        preds_ema.append(outputs_ema.detach().cpu())
        labels.append(targets.detach().cpu())
    
        # record memory
        mem.update(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % (2*args.print_freq) == 0:
            progress.display(i, logger)

    # saved data
    targets = torch.cat(labels).numpy()
    preds = torch.cat(preds).numpy()
    preds_ema = torch.cat(preds_ema).numpy()

    data_regular = np.concatenate((preds, targets), axis=1)
    saved_name_regular = 'tmpdata/data_regular_tmp.{}.txt'.format(dist.get_rank())
    np.savetxt(os.path.join(args.output, saved_name_regular), data_regular)
    data_ema = np.concatenate((preds_ema, targets), axis=1)
    saved_name_ema = 'tmpdata/data_ema_tmp.{}.txt'.format(dist.get_rank())
    np.savetxt(os.path.join(args.output, saved_name_ema), data_ema)
    if dist.get_world_size() > 1:
        dist.barrier()

    if dist.get_rank() == 0:
        logger.info("Calculating mAP:")
        filenamelist_regular = ['tmpdata/data_regular_tmp.{}.txt'.format(ii) for ii in range(dist.get_world_size())]
        
        mAP_score, APs = sl_mAP([os.path.join(args.output, _filename) for _filename in filenamelist_regular], args.num_classes)

        filenamelist_ema = ['tmpdata/data_ema_tmp.{}.txt'.format(ii) for ii in range(dist.get_world_size())]
        mAP_score_ema, APs_ema= sl_mAP([os.path.join(args.output, _filename) for _filename in filenamelist_ema], args.num_classes)

        logger.info("mAP score regular {:.4f}, mAP score EMA {:.4f}".format(mAP_score, mAP_score_ema))
    else:
        mAP_score = 0
        APs = 0
        mAP_score_ema = 0
        APs_ema = 0


    return mAP_score, APs, mAP_score_ema, APs_ema


if __name__ == '__main__':
    main()