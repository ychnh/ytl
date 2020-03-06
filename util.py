import dill as pickle
import math
import datetime
from torch import tensor
import sys
import numpy as np
import torch
import time
import os
from functools import reduce
import operator

class timebool:
    def __init__(self, timebox):
        self.timebox = timebox
        self.i, self.j = 0,0

    def set(self,i,j):
        self.i, self.j = i, j

    def reset(self):
        self.i, self.j = 0,0

    def __bool__(self):
        i,j, = self.i, self.j
        t = self.timebox[0]
        if i<=t and t<j:
            return True
        else:
            return False

from copy import deepcopy
class udict(dict):
    def __init__(self, mapping=(), def_val=None, **kwargs):
        super(udict, self).__init__(mapping, **kwargs)
        self.def_val = def_val
    def __getitem__(self, k):
        if k in self:
            return super(udict, self).get(k)
        else:
            dv= deepcopy(self.def_val)
            self[k] = dv
            return dv

def prod(L):
    return reduce(operator.mul, L, 1)

from functools import reduce
def progress_print(*text):
    stradd = lambda a,b: str(a)+' '+str(b) 
    text = reduce(stradd, text)
    print(' '*len(text), end='\r')
    print(text, end='\r')


'''
def hook_fn(m, i, o):
            print(m)
            print("------------Input Grad------------")
            for grad in i:
                try:
                    print(grad.shape)
                except AttributeError: 
                    print ("None found for Gradient")
            print("------------Output Grad------------")
            for grad in o:  
                try:
                    print(grad.shape)
                except AttributeError: 
                    print ("None found for Gradient")
                print("\n")        
        #model.register_backward_hook(hook_fn)
'''

def listdir(path,sort_time=True, files_only=False):
    if files_only:
        X = [s for s in os.listdir(path) if os.path.isfile(os.path.join(path, s))]
    else:
        X = [s for s in os.listdir(path)]

    if sort_time:
        X.sort(key=lambda s: os.path.getmtime(os.path.join(path, s)), reverse=False)

    return X


def true_path(path):
    try:
        return os.readlink(path)
    except:
        return path

# Printing
def maxmin(title,t):
    if t is not None:
        return ','.join( ['  ', title, ':', 'max', str(torch.max(t).item()), 'min', str(torch.min(t).item()) ] )
    else:
        return ','.join( ['  ', title, ':', ''])

# Directory
def check_and_make_dir(directory, ensure_empty=False):
    if not os.path.exists(directory):
        os.makedirs(directory)

    if ensure_empty:
        assert len(os.listdir(directory))==0, 'New directory needs to be empty'

# Load Save
def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name, default=None):
    try:
        if os.path.exists(name):
            with open(name, 'rb') as f:
                return pickle.load(f)
        else:
            return default
    except Exception as e:
        print(e)
        return default

def summary(model):
    for param in model.state_dict():
        print(param, '\t', model.state_dict()[param].size() )

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    st_dict = torch.load(path)
    model.load_state_dict(st_dict, strict=False)
    return model

##########################################
# Metrics

def iou(outputs: torch.Tensor, labels: torch.Tensor):
    ''' Expects the output BatchxHxW'''
    SMOOTH = 1e-4
    intersection = (outputs & labels).float().sum((1, 2))  # Will be 1 if output and label==1
    union = (outputs | labels).float().sum((1, 2))         # Will be 1 if output or label == 1
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    return iou.mean().item() #mean across batch

def pixel_accuracy(pred, labels):
    SMOOTH = 1e-4
    inter = (pred==labels)
    V = np.prod(inter.shape)
    acc = inter.sum()/V
    return acc.item()

def composition(label, num_classes):
    comp = []
    for cls in range(num_classes):
        comp.append( torch.sum(label==cls) )
    return comp

# IOU per class
# Can construct confusion matrix here
def pixel_accuracy_classes(A, B, n_classes):
    SMOOTH = 1e-4
    accuracies = []
    for c in range(n_classes):
        cA = onehot(A,c)
        cB = onehot(B,c)
        cOr = (cA | cB)
        cAnd = (cA & cB)
        acc = (cAnd.sum()+SMOOTH)/(cOr.sum()+SMOOTH)
        acc = round(acc.item(), 2)
        accuracies += [acc]

    return accuracies

class ConfusionMat():
    def __init__(self):
        self.initialized = False
        self.Mat = None

    
    def _cmat(self, A,B, n_classes):
        #SMOOTH = 1e-4

        # A is prediction and B is gt
        conf_mat = []
        for r in range(n_classes):
            rlabel = onehot(A,r)
            blabel = onehot(B,r)

            #rtotal = torch.sum(rlabel)
            mask = B.clone()
            mask[~rlabel]=-1
            # So we are counting out of our predictions what the gt was in
            row = []
            for c in range(n_classes):
                rmask = (mask==c)
                #score = ( torch.sum(rmask) )/( rtotal + SMOOTH )
                score = torch.sum(rmask)
                row.append(score)

            conf_mat.append(row)
        conf_mat = torch.LongTensor(conf_mat)
        #M = torch.max(conf_mat)
        #conf_mat = conf_mat/M
        return conf_mat


    def addBatch(self, A,B, n_classes):
        if self.initialized:
            self.Mat += self._cmat(A,B,n_classes)
        else:
            self.Mat = self._cmat(A,B,n_classes)
            self.initialized = True

    def precision(self):
        #/TP+FP (COL SUM)
        pass

    def recall(self):

        #/TP+FN (ROW SUM)
        recall = self.Mat.clone().float()
        _,n_classes = self.Mat.shape
        for r in range(n_classes):
            sumrow = torch.sum(self.Mat[r,:])
            if( sumrow!=0):
                recall[r,:]/=sumrow

        return recall

    def display(self, mat, keys):
        #indent = 'array(['

        np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
        _,n_classes = mat.shape
        for r in range(n_classes):
            row = mat[r,:].numpy()
            print(row, keys[r])
        print('  '+', '.join(keys))
        np.set_printoptions()



##########################################
# Tensor Util
def tensor_round(tnsr, n_digits):
    rounded = torch.round(tnsr* 10**n_digits) / (10**n_digits)
    return rounded


##########################################
# Image output 

def onehot(A, clss):
    OH = (A==clss)
    return OH.bool()

def B1HW_to_B3HW(batch):
    return torch.cat([batch, batch, batch], 1)

def probs_to_labels(probs):
    ''' Probs is in format BCHW '''
    p_label = torch.argmax(probs, dim=1) # unsqueeze channel index
    return p_label

def labels_to_images(labels, num_classes):
    c1image = (labels*255//num_classes).unsqueeze(1)
    c3image = B1HW_to_B3HW(c1image)
    return c3image.float()


##########################################
# Printing Functions

def print_min_max(title, tensor):
    print(title, '       ' ,'min:', torch.min(tensor).item(),'max:', torch.max(tensor).item() )


##########################################
# Printing Functions
def enforce_cuda():
    if not torch.cuda.is_available():
        print('CUDA devices not available. Exiting')
        sys.exit()

##########################################

class Stopwatch():
    def __init__(self):
        self.prev_time = time.time()

    def lap(self):
        now = time.time()
        elapsed = now - self.prev_time
        self.prev_time = now
        return elapsed

class Progress():
    def __init__(self, opt, dataloader):
        self.epoch_len = len(dataloader)
        num_epochs = opt.n_epochs

        self.batches_total = len(dataloader)*num_epochs
        
    def batches_done(self, epoch, batch):
        return self.epoch_len*epoch + batch

    def batches_remain(self, epoch, batch):
        batches_done = self.batches_done(epoch, batch)
        return self.batches_total - batches_done

    def time_remain(self, batch_time, epoch, batch):
        sec_remain = self.batches_remain(epoch, batch) * batch_time
        return datetime.timedelta(seconds=sec_remain)
