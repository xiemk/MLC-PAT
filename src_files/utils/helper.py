import os
import argparse
import random
import numpy as np
from copy import deepcopy
from PIL import Image, ImageDraw
from collections import OrderedDict

import torch
from torchvision import datasets as datasets
import torchvision.transforms as transforms
from pycocotools.coco import COCO
from sklearn.metrics import average_precision_score
from randaugment import RandAugment

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def add_weight_decay(model, weight_decay=1e-4, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]

def get_raw_dict(args):
    """
    return the dicf contained in args.
    
    e.g:
        >>> with open(path, 'w') as f:
                json.dump(get_raw_dict(args), f, indent=2)
    """
    if isinstance(args, argparse.Namespace): 
        return vars(args)   
    else:
        raise NotImplementedError("Unknown type {}".format(type(args)))

def clean_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:7] == 'module.':
            k = k[7:]  # remove `module.`
        new_state_dict[k] = v
    return new_state_dict


def function_mAP(targs, preds):
    """Returns the model's average precision for each class
    Return:
        ap (FloatTensor): 1xK tensor, with avg precision for each class k
    """

    average_precision_list = []

    for j in range(preds.shape[1]):
        average_precision_list.append(compute_avg_precision(targs[:, j], preds[:, j]))

    return 100.0 * float(np.mean(average_precision_list)), 100.0*np.array(average_precision_list)


def compute_avg_precision(targs, preds):
    
    '''
    Compute average precision.
    
    Parameters
    targs: Binary targets.
    preds: Predicted probability scores.
    '''
    
    check_inputs(targs,preds)
    
    if np.all(targs == 0):
        # If a class has zero true positives, we define average precision to be zero.
        metric_value = 0.0
    else:
        metric_value = average_precision_score(targs, preds)
    
    return metric_value


def check_inputs(targs, preds):
    
    '''
    Helper function for input validation.
    '''
    
    assert (np.shape(preds) == np.shape(targs))
    assert type(preds) is np.ndarray
    assert type(targs) is np.ndarray
    assert (np.max(preds) <= 1.0) and (np.min(preds) >= 0.0)
    assert (np.max(targs) <= 1.0) and (np.min(targs) >= 0.0)
    assert (len(np.unique(targs)) <= 2)

class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.9997, device=None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()

        # import ipdb; ipdb.set_trace()

        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)
    

class CutoutPIL(object):
    def __init__(self, cutout_factor=0.5):
        self.cutout_factor = cutout_factor

    def __call__(self, x):
        img_draw = ImageDraw.Draw(x)
        h, w = x.size[0], x.size[1]  # HWC
        h_cutout = int(self.cutout_factor * h + 0.5)
        w_cutout = int(self.cutout_factor * w + 0.5)
        y_c = np.random.randint(h)
        x_c = np.random.randint(w)

        y1 = np.clip(y_c - h_cutout // 2, 0, h)
        y2 = np.clip(y_c + h_cutout // 2, 0, h)
        x1 = np.clip(x_c - w_cutout // 2, 0, w)
        x2 = np.clip(x_c + w_cutout // 2, 0, w)
        fill_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img_draw.rectangle([x1, y1, x2, y2], fill=fill_color)

        return x


class CocoDetection(datasets.coco.CocoDetection):
    def __init__(self, root, annFile, transform=None, target_transform=None):
        self.root = root
        self.coco = COCO(annFile)

        self.ids = list(self.coco.imgToAnns.keys())
        self.transform = transform
        self.target_transform = target_transform
        self.cat2cat = dict()
        for cat in self.coco.cats.keys():
            self.cat2cat[cat] = len(self.cat2cat)
        # print(self.cat2cat)

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        output = torch.zeros((3, 80), dtype=torch.long)
        for obj in target:
            if obj['area'] < 32 * 32:
                output[0][self.cat2cat[obj['category_id']]] = 1
            elif obj['area'] < 96 * 96:
                output[1][self.cat2cat[obj['category_id']]] = 1
            else:
                output[2][self.cat2cat[obj['category_id']]] = 1
        target = output

        path = coco.loadImgs(img_id)[0]['file_name']
        # import ipdb; ipdb.set_trace()
        path_list = path.split('_')
        path = os.path.join(path_list[1], path)
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        
        
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

class TransformPatch_Train(object):
    def __init__(self, args):
        self.n_grid = args.n_grid
        self.image_size = args.image_size

        self.strong = transforms.Compose([
                    transforms.Resize((args.image_size, args.image_size)),
                    CutoutPIL(cutout_factor=0.5),
                    RandAugment(),
                    transforms.ToTensor(),
                    # normalize,
                ])

    def __call__(self, img):
        strong_list = [self.strong(img)] 
        
        # Append patches
        img = img.resize((self.image_size, self.image_size))

        # To permute the orders of patches
        x_order = np.random.permutation(self.n_grid)
        y_order = np.random.permutation(self.n_grid)

        grid_size_x = img.size[0] // self.n_grid
        grid_size_y = img.size[1] // self.n_grid
        
        for i in x_order:
            for j in y_order:
                x_offset = i * grid_size_x
                y_offset = j * grid_size_y
                patch = img.crop((x_offset, y_offset, x_offset + grid_size_x, y_offset + grid_size_y))
                # Append patches
                strong_list.append(self.strong(patch))
        
        
       
        return strong_list


class TransformPatch_Val(object):
    def __init__(self, args):
        self.n_grid = args.n_grid
        self.image_size = args.image_size
        
        self.weak = transforms.Compose([
                        transforms.Resize((args.image_size, args.image_size)),
                        transforms.ToTensor(),
                        # normalize, # no need, toTensor does normalization
                    ])

    def __call__(self, img):
        weak_list = [self.weak(img)]

        # Append patches
        img = img.resize((self.image_size, self.image_size))

        # To permute the order for local patched
        x_order = np.random.permutation(self.n_grid)
        y_order = np.random.permutation(self.n_grid)

        grid_size_x = img.size[0] // self.n_grid
        grid_size_y = img.size[1] // self.n_grid
        
        for i in x_order:
            for j in y_order:
                x_offset = i * grid_size_x
                y_offset = j * grid_size_y
                patch = img.crop((x_offset, y_offset, x_offset + grid_size_x, y_offset + grid_size_y))
                # Append patches
                weak_list.append(self.weak(patch))
        
        return weak_list

def voc_ap(rec, prec, true_num):
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def sl_mAP(imagessetfilelist, num):
    if isinstance(imagessetfilelist, str):
        imagessetfilelist = [imagessetfilelist]
    lines = []
    for imagessetfile in imagessetfilelist:
        with open(imagessetfile, 'r') as f:
            lines.extend(f.readlines())
    
    seg = np.array([x.strip().split(' ') for x in lines]).astype(float)
    gt_label = seg[:,num:].astype(np.int32)
    num_target = np.sum(gt_label, axis=1, keepdims = True)
    threshold = 1 / (num_target+1e-6)

    predict_result = seg[:,0:num] > threshold


    sample_num = len(gt_label)
    class_num = num
    tp = np.zeros(sample_num)
    fp = np.zeros(sample_num)
    aps = []
    per_class_recall = []

    for class_id in range(class_num):
        confidence = seg[:,class_id]
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        sorted_label = [gt_label[x][class_id] for x in sorted_ind]

        for i in range(sample_num):
            tp[i] = (sorted_label[i]>0)
            fp[i] = (sorted_label[i]<=0)
        true_num = 0
        true_num = sum(tp)
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(true_num)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, true_num)
        aps += [ap]

    np.set_printoptions(precision=6, suppress=True)
    mAP = np.mean(aps)

    return mAP*100, np.array(aps)*100


def calculate_metric(preds, labels):

    # N_c_k = (preds*labels).sum(0)
    # N_p_k = preds.sum(0)
    # N_g_k = labels.sum(0)

    # CP = (N_c_k/N_p_k).mean()
    # CR = (N_c_k/N_g_k).mean()
    # CF1 = 2*(CP*CR)/(CP+CR)

    # OP = N_c_k.sum()/N_p_k.sum()
    # OR = N_c_k.sum()/N_g_k.sum()
    # OF1 = 2*(OP*OR)/(OR+OR)

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

def label_decision(probs, pos_ratio):

    indices = [int(x)-1 for x in pos_ratio*probs.shape[0]]
    sorted_outputs = -np.sort(-probs, axis=0)
    thre_vec = sorted_outputs[indices, range(probs.shape[1])]
    preds = (probs.copy()>=thre_vec).astype(np.float32)

    # import ipdb; ipdb.set_trace()

    # top_thre_vec = sorted_outputs[:, 2, np.newaxis]
    # top_preds = (probs.copy()>=top_thre_vec).astype(np.float32)

    outputs_tensor = torch.from_numpy(probs).cuda()
    topk = torch.topk(outputs_tensor, 3)
    top_preds = torch.zeros_like(outputs_tensor).cuda().scatter(1,topk.indices,1).cpu().numpy()

    return preds, top_preds


def sl_mAP_cf1_of1(imagessetfilelist, num, pos_ratio):
    if isinstance(imagessetfilelist, str):
        imagessetfilelist = [imagessetfilelist]
    lines = []
    for imagessetfile in imagessetfilelist:
        with open(imagessetfile, 'r') as f:
            lines.extend(f.readlines())

    # import ipdb; ipdb.set_trace()
    
    seg = np.array([x.strip().split(' ') for x in lines]).astype(float)
    gt_label = seg[:,num:].astype(np.int32)
    num_target = np.sum(gt_label, axis=1, keepdims = True)
    threshold = 1 / (num_target+1e-6)

    predict_result = seg[:,0:num] > threshold


    sample_num = len(gt_label)
    class_num = num
    tp = np.zeros(sample_num)
    fp = np.zeros(sample_num)
    aps = []
    per_class_recall = []

    for class_id in range(class_num):
        confidence = seg[:,class_id]
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        sorted_label = [gt_label[x][class_id] for x in sorted_ind]

        for i in range(sample_num):
            tp[i] = (sorted_label[i]>0)
            fp[i] = (sorted_label[i]<=0)
        true_num = 0
        true_num = sum(tp)
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(true_num)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, true_num)
        aps += [ap]

    np.set_printoptions(precision=6, suppress=True)
    mAP = np.mean(aps)

    ##############################################################################################
    # pos_ratio = gt_label.sum(0)/gt_label.shape[0]

    preds, top_preds = label_decision(seg[:,0:num], pos_ratio)

    CP, CR, CF1, OP, OR, OF1 = calculate_metric(preds, gt_label)
    top_CP, top_CR, top_CF1, top_OP, top_OR, top_OF1 = calculate_metric(top_preds, gt_label)

    return mAP*100, np.array(aps)*100, [CP*100, CR*100, CF1*100, OP*100, OR*100, OF1*100], [top_CP*100, top_CR*100, top_CF1*100, top_OP*100, top_OR*100, top_OF1*100], [seg[:, 0:num], preds, gt_label]