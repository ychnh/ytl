import numpy as np
import math

def l1(a,b):
    return abs(a-b)


def L1(A,B):
    dist = 0
    for a,b in zip(A,B):
        dist += l1(a,b)

    return dist


def l2(a,b):
        return (a-b)**2


def L2(A,B):
    sum_dist = 0
    for a,b in zip(A,B):
        sum_dist += l2(a,b)

    return math.sqrt(sum_dist)


