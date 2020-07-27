# NOTE CODE IS INCOMPLETE AS I NEED TO FIND A GENERAL WAY TO FORMULATE THESE FUNCTIONS

import torch 
from torch import tensor

M = 256//4
cC, cR = torch.zeros(M,M).cuda(), torch.zeros(M,M).cuda()
for i in range(M):
    v = -1+2*(i)/(M-1)
    cC[:,i],cR[i,:] = v,v

mpool = torch.nn.MaxPool2d(kernel_size=4, stride=4, padding=0).cuda()

def masked_softmax(vec, mask, epsilon=1e-5):
    shape = vec.shape
    B,_,_ = shape
    vec = vec.view(B,-1)
    mask = mask.view(B,-1)
    exps = torch.exp(vec)
    masked_exps = exps * mask.float()
    masked_sums = masked_exps.sum(1, keepdim=True) + epsilon
    return (masked_exps/masked_sums).view(*shape)


def get_coord_for_heat( heat, T=.2):
    B,S,_ = heat.shape
    
    heat = masked_softmax(heat, heat>T)
    R = heat* torch.stack( [cR]*6 )
    C = heat* torch.stack( [cC]*6 )

    def get_coord(M):
        #v = (M.sum()/(M!=0).sum()).item()
        v = M.sum()*S//2+S//2
        return v
    
    p = tensor([ (get_coord(R[i]), get_coord(C[i])) for i in range(B) ])
    return p

# TODO: Make a more general extendable library
def masked_softmax_general(vec, mask, dim=1, epsilon=1e-5):
    shape = vec.shape
    B,C,_,_ = shape
    vec = vec.view(B,C,-1)
    mask = mask.view(B,C,-1)
    exps = torch.exp(vec)
    masked_exps = exps * mask.float()
    masked_sums = masked_exps.sum(2, keepdim=True) + epsilon
    return (masked_exps/masked_sums).view(*shape)


