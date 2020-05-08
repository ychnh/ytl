# Visual
import math
import matplotlib.pyplot as plt
import numpy as np

def pyplot_show_grid(images, save_path=None):
    ''' Displays multiple `images` in a grid shape.
    * images: a list of numpy/PIL images
    '''
    L = len(images)
    R = math.floor( math.sqrt(L) )
    C = math.ceil(L/R)
    #print('diplaying',L, 'images:',R,'x',C)

    fig = plt.figure(figsize=(25,25))
    idx = 0
    f, axarr = plt.subplots(R,C)
    for i in range(R):
        for j in range(C):
            arrIdx = C*i+j
            if (arrIdx >= L):
                break
            else:
                axarr[i,j].imshow(images[arrIdx])
            idx+=1

    if save_path == None:
        plt.show()
    else:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

def pyplot_show_list(images):
    ''' Shows multiple large images in a single column
    * images: list of numpy/PIL images
    '''
    N = len(images)
    plt.figure(figsize=(10,25))
    for i,img in enumerate(images):
        axis = plt.subplot(N,1,i+1)
        plt.imshow(img)

def pyplot_show_images(images):
    ''' Shows multiple large images in a single column
    * images: list of numpy/PIL images
    '''
    N = len(images)
    plt.figure(figsize=(10,25))
    for i,img in enumerate(images):
        axis = plt.subplot(N,1,i+1)
        plt.imshow(img)

def pyplot_show_image(img):
    ''' Shows multiple large images in a single column
    * images: list of numpy/PIL images
    '''
    #plt.figure(figsize=(10,25))
    plt.imshow(img)

     
def pyplot_histogram(X, bins=3, ylabel='',xlabel='', normed=False):
    ret = plt.hist(X, normed=normed, bins=bins)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    return ret

def pyplot_graph(X, ylabel='', xlabel=''):
    plt.plot(X)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
