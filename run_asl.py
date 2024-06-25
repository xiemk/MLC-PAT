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
from src_files.utils.helper import function_mAP, add_weight_decay, get_raw_dict, ModelEma
from src_files.utils.losses import AsymmetricLoss
from src_files.models.factory import create_model
from src_files.data.data import get_datasets


from torch.cuda.amp import GradScaler, autocast

NUM_CLASS = {'voc2007': 20, 'voc2012': 20, 'coco': 80, 'nus': 81, 'vg500': 500,'vg80':80, 'vg200':201}

def get_args():
    parser = argparse.ArgumentParser(description='Clean ASL Training')

    # data
    parser.add_argument('--data_name', help='dataset name', default='coco', choices=['voc2007', 'voc2012', 'coco', 'nus','vg500','vg200','vg80'])
    parser.add_argument('--data_dir', help='dir of all datasets', default='/home/algroup/xmk/data')
    parser.add_argument('--image_size', default=448, type=int,
                        help='size of input images')
    parser.add_argument('--output', metavar='DIR', default='./outputs',
                        help='path to output folder')

    # model
    parser.add_argument('--model_name', default='tresnet_l')
    parser.add_argument('--model_path', default='./pretrained/tresnet_l_448.pth', type=str)
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
    parser.add_argument('--early_stop', action='store_true', help='apply early stop')
    parser.add_argument('--pct_start', default=0.2, type=float)
    parser.add_argument('--gamma_neg', default=4, type=int)
    parser.add_argument('--clip', default=0.05, type=float)

    # random seed
    parser.add_argument('--seed', default=1, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--benchmark', default=True, type=bool,
                        help='seed for initializing training. ')
    
    parser.add_argument('--gpu', default=0, type=int,
                        help='seed for initializing training. ')


    args = parser.parse_args()

    args.num_classes = NUM_CLASS[args.data_name]
    args.data_dir = os.path.join(args.data_dir, args.data_name) 
    
    # time_str = datetime.datetime.strftime(datetime.datetime.now(),'%Y%m%d%H%M%S')
    args.output = os.path.join(args.output, args.data_name, f'asl_{args.benchmark}_{args.model_name}_{args.image_size}_{args.optim}_{args.lr}_{args.weight_decay}_{args.gamma_neg}_{args.clip}_{args.pct_start}_{args.batch_size}_{args.epochs}_seed_{args.seed}')
    return args



def main():
    args = get_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = args.benchmark
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

    ema_m = ModelEma(model, args.ema_decay)  # 0.9997^641=0.82

    # Data loading
    train_dataset, val_dataset = get_datasets(args)
    logger.info("len(train_dataset)): {}".format(len(train_dataset)))
    logger.info("len(val_dataset)): {}".format(len(val_dataset)))

    # Pytorch Data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)
    
    epoch_time = AverageMeterHMS('TT')
    eta = AverageMeterHMS('ETA', val_only=True)
    mAPs = AverageMeter('mAP', ':5.5f', val_only=True)
    mAPs_ema = AverageMeter('mAP_ema', ':5.5f', val_only=True)
    progress = ProgressMeter(
        args.epochs,
        [eta, epoch_time, mAPs, mAPs_ema],
        prefix='=> Test Epoch: ')

    # Set optimizer
    # parameters = add_weight_decay(model, args.weight_decay)
    # optimizer = torch.optim.Adam(params=parameters, lr=args.lr, weight_decay=0)  # true wd, filter_bias_and_bn
    optimizer = set_optimizer(model, args)
    args.steps_per_epoch = len(train_loader)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=args.steps_per_epoch, epochs=args.epochs, pct_start=args.pct_start)

    # Set loss func
    criterion = AsymmetricLoss(gamma_neg=args.gamma_neg, gamma_pos=0, clip=args.clip, disable_torch_grad_focal_loss=True)
   
    end = time.time()
    best_epoch = -1
    best_regular_mAP = 0
    best_regular_epoch = -1
    best_ema_mAP = 0
    regular_mAP_list = []
    ema_mAP_list = []
    mAP_ema_test = 0
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
                    break

    print("Best mAP:", best_mAP)

    if summary_writer:
        summary_writer.close()
    
    return 0

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

def train(train_loader, model, ema_m, optimizer, scheduler, epoch, args, logger, criterion):
    scaler = GradScaler()
    
    losses = AverageMeter('Loss', ':5.3f')
    lr = AverageMeter('LR', ':.3e', val_only=True)
    mem = AverageMeter('Mem', ':.0f', val_only=True)
    progress = ProgressMeter(
        args.steps_per_epoch,
        [lr, losses, mem],
        prefix="Epoch: [{}/{}]".format(epoch, args.epochs))

    def get_learning_rate(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    lr.update(get_learning_rate(optimizer))
    logger.info("lr:{}".format(get_learning_rate(optimizer)))

    # switch to train mode
    model.train()

    for i, (inputs, targets) in enumerate(train_loader):

        # **********************************************compute loss*************************************************************

        inputs = inputs.cuda()
        targets = targets.cuda()
        # mixed precision ---- compute outputs
        with autocast():
            logits = model(inputs).float()
        
        loss = criterion(logits, targets)

        # record loss
        losses.update(loss.item(), inputs.size(0))
        mem.update(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)

        # compute gradient and do SGD step
        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # one cycle learning rate
        scheduler.step()
        lr.update(get_learning_rate(optimizer))
        ema_m.update(model)


        if i % args.print_freq == 0:
            progress.display(i, logger)

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
        inputs = inputs.cuda()
    
        # compute output
        with autocast():
            outputs = torch.sigmoid(model(inputs))
            outputs_ema = torch.sigmoid(ema_m.module(inputs))
        
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

    # calculate mAP
    mAP, APs= function_mAP(torch.cat(labels).numpy(), torch.cat(preds).numpy())
    mAP_ema, APs_ema= function_mAP(torch.cat(labels).numpy(), torch.cat(preds_ema).numpy())
    
    print("Calculating mAP:")  
    logger.info("  mAP: {}  mAP_EMA: {} ".format(mAP, mAP_ema))

    return mAP, APs, mAP_ema, APs_ema


if __name__ == '__main__':
    main()