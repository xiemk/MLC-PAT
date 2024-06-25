
import os
import torch
import torch.nn as nn
from torchvision import models

from .tresnet import TResnetL
from .tresnet_v2 import TResnetL_v2
from src_files.helper_functions.distributed import print_at_master
from .query2labels.models.query2label import build_q2l
from .query2labels.models.query2label import GroupWiseLinear

def get_model_path(modelname, pretrain_dir):
    """
        Config your pretrained model path here!
    """

    PTDICT = {
        'tresnetl_in1k': 'tresnet_l_448.pth',
        'tresnetl_v2_in21k': 'tresnet_l_v2_miil_21k.pth',
        'tresnetl_v2_io': 'tresnet_l_pretrain_ml_decoder.pth',
    }

    return os.path.join(pretrain_dir, PTDICT[modelname]) 

class TripleClassifier(nn.Module):
    def __init__(self, in_dim, out_dim, state_dict=None):
        super(TripleClassifier, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(in_dim, out_dim)
        self.fc3 = nn.Linear(in_dim, out_dim)

        if state_dict is not None:
            self.fc1.load_state_dict(state_dict[0])
            self.fc2.load_state_dict(state_dict[1])
            self.fc3.load_state_dict(state_dict[2])

    def forward(self, x):
        logit1 = self.fc1(x)
        logit2 = self.fc2(x)
        logit3 = self.fc3(x)
        return logit1, logit2, logit3
    
class DoubleClassifier(nn.Module):
    def __init__(self, in_dim, out_dim, state_dict=None):
        super(DoubleClassifier, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(in_dim, out_dim)

        if state_dict is not None:
            self.fc1.load_state_dict(state_dict[0])
            self.fc2.load_state_dict(state_dict[1])

    def forward(self, x):
        logit1 = self.fc1(x)
        logit2 = self.fc2(x)

        return logit1, logit2
    
class TripleQ2LHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(TripleQ2LHead, self).__init__()
        self.fc1 = GroupWiseLinear(out_dim, in_dim, bias=True)
        self.fc2 = GroupWiseLinear(out_dim, in_dim, bias=True)
        self.fc3 = GroupWiseLinear(out_dim, in_dim, bias=True)


    def forward(self, x):
        logit1 = self.fc1(x)
        logit2 = self.fc2(x)
        logit3 = self.fc3(x)
        return logit1, logit2, logit3
    
class DoubleQ2LHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DoubleQ2LHead, self).__init__()
        self.fc1 = GroupWiseLinear(out_dim, in_dim, bias=True)
        self.fc2 = GroupWiseLinear(out_dim, in_dim, bias=True)

    def forward(self, x):
        logit1 = self.fc1(x)
        logit2 = self.fc2(x)

        return logit1, logit2
    
def q2l_head(in_features, num_classes, logits_attention):

    if logits_attention == None:
        head = GroupWiseLinear(num_classes, in_features, bias=True)
    if logits_attention == 'self':
        head = DoubleQ2LHead(in_features, num_classes)
    elif logits_attention == 'cross':
        head = TripleQ2LHead(in_features, num_classes)
    
    return head

def linear_head(in_features, num_classes, logits_attention):
    
    if logits_attention == None:
        classifier = nn.Linear(in_features, num_classes)
    elif logits_attention == 'self':
        classifier = DoubleClassifier(in_features, num_classes)
    elif logits_attention == 'cross':
        classifier = TripleClassifier(in_features, num_classes)
    
    return classifier

def load_model_weights(model, model_path):
    state = torch.load(model_path, map_location='cpu')
    if 'model' in state:
        state_key = 'model'
    else:
        state_key = 'state_dict'
    for key in model.state_dict():
        if 'num_batches_tracked' in key:
            continue
        p = model.state_dict()[key]
        if key in state[state_key]:
            ip = state[state_key][key]
            if p.shape == ip.shape:
                p.data.copy_(ip.data)  # Copy the data of parameters
            else:
                print_at_master(
                    'could not load layer: {}, mismatch shape {} ,{}'.format(key, (p.shape), (ip.shape)))
        else:
            print_at_master('could not load layer: {}, not in checkpoint'.format(key))
    return model

def build_head(model, args):

    if 'logits_attention' not in args:
        args.logits_attention = None
        
    if args.model_name in ['resnet101']:
        model.fc = linear_head(model.fc.in_features, args.num_classes, args.logits_attention)
    elif args.model_name in ['tresnetl', 'tresnetl_v2']:
        model.head.fc = linear_head(model.head.fc.in_features, args.num_classes, args.logits_attention)
    elif 'q2l' in args.model_name:
        model.fc = q2l_head(model.transformer.d_model, args.num_classes, args.logits_attention)
    else:
        print("model: {} not defined !!".format(args.model_name))
        exit(-1)

    return model

def create_model(args):

    model_params = {'args': args, 'num_classes': args.num_classes}
    args.model_name = args.model_name.lower()
    
    if args.model_name == 'resnet101':
        model = models.resnet101(pretrained=True)
        args.pretrain_type = None
    elif args.model_name == 'tresnetl':
        model = TResnetL(model_params)
    elif args.model_name == 'tresnetl_v2':
        model = TResnetL_v2(model_params)
    elif 'q2l' in args.model_name:
        model = build_q2l(args)
        args.pretrain_type = None
    else:
        print("model: {} not found !!".format(args.model_name))
        exit(-1)

    if args.pretrain_type and args.pretrain_type!='' and args.resume=='':
        args.pretrain_path = get_model_path(args.model_name+'_'+args.pretrain_type, args.pretrain_dir)
        model = load_model_weights(model, args.pretrain_path)
    
    model = build_head(model, args)

    return model