import math
def mean(data):
    N = len(data)
    data_sum = 0
    for d in data:
        data_sum += d

    return data_sum/N

def std_dev(data_list, m):
    sqr_dist_sum = 0
    N = 0
    for d in data_list:
        sqr_dist_sum += (d-m)**2
        N += 1

    return math.sqrt( 1/(N-1) * sqr_dist_sum )


def unorm_d(data, m, std):
    return data*std + m

def norm_d(data, m, std):
    return (data-m) / std


def normalize_data(data):
    data_sum = 0
    m = mean(data)

    std = std_dev(data, m)
    n_data = []
    for d in data:
        n_data.append( norm_d(d, m, std) )

    return n_data, m, std

def unnormalize_data(n_data, m, std):
    data = []
    for nd in n_data:
        data.append( unorm_d(nd, m, std) )

    return data

#maxnormalizations need to change
#def normalize_data_list2(data_list):
    #M = max(data_list)
    #return [ d/M for d in data_list], M
#def unnormalize_data_list2(data_list, M):
    #return [ d*M for d in data_list]

