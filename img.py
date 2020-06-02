import cv2
from numpy import array
from numpy import pi
import numpy as np
from PIL import Image

def shift_scale_rotate( img, angle=0, dx=0, dy=0, sx=1,sy=1, center=None, interpolation=cv2.INTER_CUBIC, border_mode=cv2.BORDER_CONSTANT, value=0 ):
    height, width = img.shape[:2]
    if center is None:
        center = (width / 2, height / 2)
    else:
        r,c = center
        center = (c,r)
    scale_mat = array( [[sx,0,0],[0,sy,0],[0,0,1]] )
    matrix = cv2.getRotationMatrix2D(center, 180/pi * angle, 1)
    matrix[0, 2] += dx
    matrix[1, 2] += dy
    matrix = matrix @ scale_mat

    warp_affine_fn = _maybe_process_in_chunks(
      cv2.warpAffine, M=matrix, dsize=(width, height), flags=interpolation, borderMode=border_mode, borderValue=value
    )
    return warp_affine_fn(img), matrix

def get_num_channels(image):
    return image.shape[2] if len(image.shape) == 3 else 1

def _maybe_process_in_chunks(process_fn, **kwargs):
    """
    Wrap OpenCV function to enable processing images with more than 4 channels.
    Limitations:
        This wrapper requires image to be the first argument and rest must be sent via named arguments.
    Args:
        process_fn: Transform function (e.g cv2.resize).
        kwargs: Additional parameters.
    Returns:
        numpy.ndarray: Transformed image.
    """

    def __process_fn(img):
        num_channels = get_num_channels(img)
        if num_channels > 4:
            chunks = []
            for index in range(0, num_channels, 4):
                chunk = img[:, :, index : index + 4]
                chunk = process_fn(chunk, **kwargs)
                chunks.append(chunk)
            img = np.dstack(chunks)
        else:
            img = process_fn(img, **kwargs)
        return img

    return __process_fn


def resize(im,K):
    im = array(im)
    H,W,_ = im.shape
    im = Image.fromarray( im )
    _H,_W = int(K*H), int(K*W) 
    return array( im.resize( (_W,_H), Image.LANCZOS) )

