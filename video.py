from os import listdir
from os.path import join
from PIL import Image
import cv2
import torch
import numpy as np


def minsec(frm, fps=30):
    return ( frm/fps ) // 60, ( frm/fps ) % 60

def frm(m,s,fps=30):
    return int( fps*(m*60+s) )

class ImageDirectory:
    def __init__(self, path, ext='.png', chan=True):
        self.chan = chan

        ispic = lambda x: x[-4:]=='.png'
        comb = lambda x: join(path, x)

        files = listdir(path)
        files.sort()
        files = filter( ispic, files )
        files = map(comb, files)
        self.files = list(files)

    def __getitem__(self, key):
        frames = []
        key_len = (len(key)==4 and self.chan) or (len(key)==3 and not self.chan)
        
        if (isinstance(key, tuple) and key_len):
            if self.chan: k_frame, k_R, k_C, k_chan = key
            else: k_frame, k_R, k_C = key

            frm_slice = isinstance(k_frame, slice)
            if frm_slice: I = range(k_frame.start, k_frame.stop)
            else: I = [k_frame]

            for i in I:
                img = self.files[i]
                img = Image.open( img )
                img = torch.ByteTensor( np.array(img) )
                if self.chan: img = img[k_R, k_C, k_chan]
                else: img = img[k_R, k_C]

                frames.append(img)

        if frm_slice: return frames
        else: return frames[0]

    def __len__(self):
        return len(self.files)

        

from time import time

class ImageSequence:
    def __init__(self, path):
        self.path = path
        
        cap = cv2.VideoCapture(self.path)
        self.length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def __getitem__(self, key):
        sequence = []
        if ( isinstance(key, tuple) and len(key)==4  ):
            #s_frame, s_chan, s_R, s_C = key
            s_frame, s_R, s_C, s_chan = key
            if isinstance(s_frame, slice):
                start, stop = s_frame.start, s_frame.stop
            else:
                start, stop = s_frame, s_frame+1
            
            cap = cv2.VideoCapture(self.path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)

            for i in range(start, stop):
                ret, frame = cap.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame[s_R, s_C, s_chan].astype('uint8')
                sequence.append( torch.ByteTensor(frame))              


        else:
            print('Invalid get parameters. Need 4 slices for batch, channel, R, C')                     
            print(key)
        
        ret_img = torch.stack(sequence)

        if not isinstance(s_chan, slice): 
            ret_img = ret_img.squeeze(2)
        if not isinstance(s_frame, slice):
            return ret_img.squeeze(0)

        return ret_img
    
    def __len__(self):
        return self.length

