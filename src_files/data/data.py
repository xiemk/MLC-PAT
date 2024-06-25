from cProfile import label
import os
import random
import numpy as np
import torch
from randaugment import RandAugment
import torchvision.transforms as transforms
from PIL import ImageDraw
from src_files.data.handlers import COCO2014_handler, VOC2007_handler, VG256_handler

np.set_printoptions(suppress=True)

HANDLER_DICT = {
    'voc2007': VOC2007_handler,
    'coco': COCO2014_handler,
    'vg256': VG256_handler
}

def get_datasets(args, patch=False):

    if patch:
        train_transform = TransformPatch_Train(args)
        val_transform = TransformPatch_Val(args)

    else:
        train_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        CutoutPIL(cutout_factor=0.5),
        RandAugment(),
        transforms.ToTensor()])

        val_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor()])

    # load data:
    source_data = load_data(args.data_dir)
	
    data_handler = HANDLER_DICT[args.data_name]

    train_dataset = data_handler(source_data['train']['images'], source_data['train']['labels'], args.data_dir, transform=train_transform)
    
    val_dataset = data_handler(source_data['val']['images'], source_data['val']['labels'], args.data_dir, transform=val_transform)

    return train_dataset, val_dataset



def load_data(base_path):
    data = {}
    for phase in ['train', 'val']:
        data[phase] = {}
        data[phase]['labels'] = np.load(os.path.join(base_path, 'formatted_{}_labels.npy'.format(phase)))
        data[phase]['images'] = np.load(os.path.join(base_path, 'formatted_{}_images.npy'.format(phase)))
    return data

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


class TransformPatch_Train(object):
    def __init__(self, args):
        self.n_grid = args.n_grid
        self.image_size = args.image_size

        self.strong = transforms.Compose([
                    transforms.Resize((args.image_size, args.image_size)),
                    CutoutPIL(cutout_factor=0.5),
                    RandAugment(),
                    transforms.ToTensor(),
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




