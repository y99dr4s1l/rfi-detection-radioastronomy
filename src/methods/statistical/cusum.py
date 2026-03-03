import numpy as np

def CUSUM(arr, k, h, output=False):
    #arr = data.to_numpy()
    #arr = data
    C = np.zeros(shape=arr.shape)
    mask = np.zeros(shape=arr.shape)
    for t in range(1, arr.shape[1]):
        c = C[:, t-1] + arr[:, t] - k
        c[c<0] = 0
        C[:, t] = c
        mask[:, t][C[:, t] > h] = 1
        C[:, t][C[:, t] > h] = 0
    if output == False:
        return mask
    else:
        return C
        #return C, mask, (h, k)    

