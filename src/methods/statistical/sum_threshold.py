import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve1d
from tqdm import tqdm
import scipy.ndimage as im

def chi_threshold5(i, chi1, target_value, acc=0):
    chi = {}
    decay_factor = 1 / np.sqrt(2)
    for iteration in range(1, i+1):
        chi[2**iteration] = ((chi1 - target_value) * ((decay_factor*(1-acc)) ** np.log2(2**iteration + 1)))
    return chi

def sumthreshold(arr, chi, output=False):
    #print(arr)
    #plt.imshow(arr[2300:2400, 250:350])
    #plt.show()
    mask = np.zeros(shape=arr.shape, dtype='bool')
    S = {}

    for key in chi.keys():
        w = key
        t = chi[w] * w
        new_arr = arr.copy()
        flag = mask.copy()
        new_arr[mask] = chi[w]
        kernel_m = np.ones((1, w))
        sum_arr = im.convolve(new_arr, kernel_m, mode='constant', cval=0, origin=0)
        sum_arr = sum_arr[:, int(w/2)-1:-int(w/2)]
        S[w] = sum_arr
        flag[:, int(w/2)-1:-int(w/2)][sum_arr > t] = True
        #print(f't: x < {t}')
        #print(sum_arr)
        #plt.imshow(sum_arr[2300:2400, 250:350])
        #plt.show()
        for i in range(1, w):
            flag = flag + np.roll(flag, 1, axis=1)
        flag = flag > 0
        mask = flag + mask
        #plt.imshow(flag[2300:2400, 250:350])
        #plt.show()
        #plt.imshow(mask[2300:2400, 250:350])
        #plt.show()
     
    if output == True:
        return mask, S, chi
    else:
        return mask