import os
import pickle as pick
#To Use the file add the following to the top of the python file
'''
---------------------------------------
Syspath import snippet
---------------------------------------
import sys
sys.path.append('/home/yhong')


---------------------------------------
Argument Parsing Snippet
---------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, help="path to images directory")
args = vars(ap.parse_args())
'''

#TODO: Make this into a global module available in pip3 or atelast have a setup.py
def get_file_dir(file):
    '''
    Inputs to this function will be __file__
    '''
    return os.path.dirname(file) +'/'

def traverse_dir_and_do_func(cwd, dirs, func):
    '''
    passes params to func,
    cwd, idx, dir_name
    '''
    for i,d in enumerate(dirs):
        d_path = os.path.join(cwd, d)
        func(cwd=cwd, idx=i, dir_name=d)

def mkdir(dirname):
    try:
        os.mkdir(imgt_save_dir)
    except:
        print(dirname, 'exists')



def writeListToFile(writeList,filename):
    with open(filename, 'w') as f:
        for item in writeList:
            f.write("%s\n" % item)

def writeFileToList(filename):
    returnList = []
    with open(filename, 'r') as filehandle:
        for line in filehandle:
            line = line.rstrip('\n')
            returnList.append(line)
    return returnList

def unpickle(file):
    with open(file, 'rb') as fo:
        data = pick.load(fo, encoding='bytes')
    return data

def pickle(data,filename):
    f = open(filename,"wb")
    pick.dump(data,f,protocol=2)
    f.close()

def printFileByLine(filename):
    f = open(filename, 'r')
    line = f.readline()
    while line:
        line = f.readline()
    f.close()


'''
def fixEncoding(filename):
    with codecs.open(filename, 'r', encoding='iso-8859-15') as f:
        text = f.read()
    # process Unicode text
    with codecs.open(filename, 'w', encoding='utf8') as f:
        f.write(text)
'''

def filesInDir(dir):
    files = os.listdir(dir)
    return files


def printArray(array):
    for a in array:
        print(a)
        print()

import math
import sys
import random as rand
import numpy as np

def k_means(data_list, k, means=None, iterations=40):
    '''
    if  means is provided, its length must be equalto k
    '''
    # if means is empty initialize k_means with values from the list
    rand.shuffle(data_list)
    if means is None:
        means = [ data_list[i] for i in range(k)]

    def dist(A,B):
        #R2 EUC Dist
        sq_sum = 0
        for i,a in enumerate(A):
            b = B[i]
            sq_sum += (a - b)**2
        return math.sqrt(sq_sum)


    def k_nearest_partiton(data_list, means):
        k = len(means)
        S = [ [] for i in range(k) ]

        for d in data_list:
            min_dist = sys.float_info.max
            min_dist_group_idx = 0

            for i,m in enumerate(means):
                l = dist(d,m)
                if l < min_dist:
                    min_dist = l
                    min_dist_group_idx = i

            S[min_dist_group_idx].append(d)

        return S

    def eval_means(S):
        means = []
        for s in S:
            Ns = 0
            midpoint = np.zeros(s[0].shape)
            for p in s:
                midpoint = midpoint+p
                Ns += 1
            midpoint = midpoint/Ns
            means.append(midpoint)
        return means

    for i in range(iterations):
        print('k_means', i, 'iteration')
        print(means)
        S = k_nearest_partiton(data_list, means)
        means = eval_means(S)
    return means
