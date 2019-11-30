import torch
import numpy as np
import torch.nn as nn
import torchvision.models.resnet as rn
import torch.nn.functional as F
import torchvision.models.segmentation.deeplabv3 as dlabv3
from importlib import reload
reload(dlabv3)

def summary_model(model):
    for param in model.state_dict():
        print(param, '\t', model.state_dict()[param].size() )
def load_state_dict(model, path):
    st_dict = torch.load(path)
    model.load_state_dict(st_dict, strict=False)
def save_state_dict(model, path):
    torch.save_state_dict(model.state_dict(), path)

def to_img(tensor):
    ''' given a normalized tensor of CHW returns a numpy array of the image'''
    #IN = ((INPUT[0,:,:,:].permute(1,2,0)*.5+.5)*255).to( 'cpu', torch.uint8).numpy()
    x = tensor.permute(1,2,0) # CHW -> HWC
    x = (x*.5+.5) * 255 # Unormalize
    x = x.to('cpu', torch.uint8).numpy()
    return x

def to_tensor(img):
    ''' given a numpy image, first convert it to 0..1 range and then normalize using std and mean of both .5 '''
    transforms_ = [
        transforms.ToTensor(), # This will convert HWC to CHW also ranges the values between 0 and 1
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # Normalize mean and std for each channel
    ]
    tf = transforms.Compose(transforms_)
    return tf(img)

def iou(outputs: torch.Tensor, labels: torch.Tensor):
    ''' Expects the output BatchxHxW'''
    SMOOTH = 1e-4
    intersection = (outputs & labels).float().sum((1, 2))  # Will be 1 if output and label==1
    union = (outputs | labels).float().sum((1, 2))         # Will be 1 if output or label == 1
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    return iou.mean().item() #mean across batch

def m_iou(predictions: torch.Tensor, labels: torch.Tensor, weights = None):
    ''' predictions BxCxHxW '''
    C = torch.max(predictions).item()
    if weights == None:
        weights = [1/(C+1)] *(C+1)

    ious = []
    for c in range(C):
        one_mask = onehot_for_c(predictions, c)
        one_labels = onehot_for_c(labels, c)
        #ious += [ weights[c] * iou(one_mask, one_labels) ]
        ious += [ round(iou(one_mask, one_labels) ,2) ]
    return sum(ious)/len(ious)

def pixel_accuracy(pred, labels):
    #A = pred+1
    #B = pred+1
    inter = (pred==labels)

    V = np.prod(inter.shape)
    return (inter.sum()/V).item()

def onehot_for_c(predictions, c):
    pre_mask = (predictions+1) # shift the range of items from 1-n
    k = c+1 # projection of c 
    mask = (pre_mask.clone())
    mask[mask!=(k)] = 0 # make all non-k values zero
    return mask
