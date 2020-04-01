import numpy as np
from jupyter_innotater import *
import numpy as np, os
import glob
from jupyter_innotater import *
import numpy as np 
from ytil import util
from os.path import join

class Annotator:
    def __init__(self, filepaths, rcboxes=None, rcpoints=None, width=500, height=500):
        ''' box or points are a (L,F,4) np.array '''
        assert not (rcboxes is None and rcpoints is None)
        if rcboxes is not None:
            self.set_boxes(rcboxes)
        if rcpoints is not None:
            self.set_points(rcpoints)

        self.inn = build_innotater(filepaths, self._crboxes, width, height)

    def set_boxes(self, rcboxes):
        self._crboxes = crboxes(rcboxes)

    def boxes(self):
        return rcboxes( self._crboxes )

    def set_points(self, rcpoints):
        crpoints = crboxes(rcpoints)
        self._crboxes = blowpoints(crpoints)

    def points(self):
        return shrinkpoints( rcboxes( self._crboxes ) )

    def show(self):
        return self.inn


def build_innotater(filepaths, boxes, width, height):
    repeats = boxes.shape[1]
    inn = Innotater(
        [
            ImageInnotation(filepaths, path='', width=width, height=height), # Display the image itself
            TextInnotation(filepaths, multiline=False) # Display the image filename
        ],
        [
            #BinaryClassInnotation(targets_exclude, name='Exclude'), # Checkbox
            RepeatInnotation(
                (BoundingBoxInnotation, boxes), # Individual animal bounding box
                max_repeats=repeats, min_repeats=repeats
            )
        ])
    return inn


S = 20
def shrinkpoints(bpoints):
    _shrinkpoints = np.zeros( bpoints.shape ) 
    L = bpoints.shape[0]
    N = bpoints.shape[1]
    for l in range(L):
        for n in range(N):
            a,b,al,bl = bpoints[l,n,:]
            _shrinkpoints[l,n,:] = [a+S//2,b+S//2,0,0]
    return _shrinkpoints

def blowpoints(points):
    _blowpoints = np.zeros( points.shape ) 
    L = points.shape[0]
    N = points.shape[1]
    for l in range(L):
        for n in range(N):
            a,b,al,bl = points[l,n,:]
            _blowpoints[l,n,:] = [a-S//2,b-S//2,S,S]
    return _blowpoints

def rcboxes(crboxes):
    _rcboxes = np.zeros( crboxes.shape ) 
    L = crboxes.shape[0]
    N = crboxes.shape[1]
    for l in range(L):
        for n in range(N):
            c,r,cl,rl = crboxes[l,n,:]
            _rcboxes[l,n,:] = [r,c,rl,cl]
    return _rcboxes

def crboxes(rcboxes):
    _crboxes = np.zeros( rcboxes.shape )
    L = rcboxes.shape[0]
    N = rcboxes.shape[1]
    for l in range(L):
        for n in range(N):
            r,c,rl,cl = rcboxes[l,n,:]
            _crboxes[l,n,:] = [c,r,cl,rl]
    return _crboxes
