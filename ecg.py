import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import resample
from scipy.io import savemat
from collections import Counter
from tqdm import tqdm

def read_img(f):
    im = cv2.imread(f, 0)
    im = cv2.resize(im, (2200, 1600))
    _, im = cv2.threshold(im, 128, 255, cv2.THRESH_OTSU)
    return im

def histogram(arr, axis=0):
    hist = []
    if axis == 0:
        for i in range(arr.shape[axis]):
            hist.append(len(np.where(arr[i, :]==0)[0]))
    if axis == 1:
        for i in range(arr.shape[axis]):
            hist.append(len(np.where(arr[:, i]==0)[0]))
    return hist

def find_baseline(im):
    hist = histogram(im, 0)
    bl0 = np.argmax(hist[300:600]) + 300
    bl1 = np.argmax(hist[600:900]) + 600
    bl2 = np.argmax(hist[900:1200]) + 900
    bl3 = np.argmax(hist[1200:1500]) + 1200
    return [bl0, bl1, bl2, bl3]

def noise_removal(ll):
    for i in range(1, ll.shape[0]-1):
        for j in range(1, ll.shape[1]-1):
            arr = ll[i-1:i+2, j-1:j+2]
            if len(np.where(arr==0)[0]) <= 1:
                ll[i, j] = 255
    return ll

def find_ref_pulse(im, bl0):
    ref_pulse = im[bl0-100 : bl0+15, 80:140]
    ref_pulse = noise_removal(ref_pulse)
    xmax, xmin = 17, 0
    for i in range(ref_pulse.shape[1]):
        if ref_pulse[60, i] == 0:
            xmin = i + 1
            break
    for i in range(ref_pulse.shape[1]-1, 0, -1):
        if ref_pulse[60, i] == 0:
            xmax = i - 1
            break
    w = xmax - xmin
    h = int(4.6*w)
    return ref_pulse, [w, h]

def detact_leads(im, bl):
    blt = bl[0] - 180
    im = im[blt : bl[-1]+180, 160:2100]
    bl[0] = 180; bl[1] = bl[1] - blt; bl[2] = bl[2] - blt; bl[3] = bl[3] - blt
    wl = int(im.shape[1]/4)
    
    l0 = im[:bl[0]+150, 8 : wl-8]  #; l0 = noise_removement(l0)
    l1 = im[bl[1]-150 : bl[1]+150, 8 : wl-8] #; l1 = noise_removement(l1)
    l2 = im[bl[2]-150 : bl[2]+150, 8 : wl-8]

    l3 = im[:bl[0]+150, wl+8 : 2*wl-8]
    l4 = im[bl[1]-150 : bl[1]+150, wl+8 : 2*wl-8]
    l5 = im[bl[2]-150 : bl[2]+150, wl+8 : 2*wl-8]

    l6 = im[:bl[0]+150, 2*wl+8 : 3*wl-8]
    l7 = im[bl[1]-150 : bl[1]+150, 2*wl+8 : 3*wl-8]
    l8 = im[bl[2]-150 : bl[2]+150, 2*wl+8 : 3*wl-8]

    l9 = im[:bl[0]+150, 3*wl+8 : im.shape[1]-8]
    l10 = im[bl[1]-150 : bl[1]+150, 3*wl+8 : im.shape[1]-8]
    l11 = im[bl[2]-150 : bl[2]+150, 3*wl+8 : im.shape[1]-8]

    l12 = im[bl[3]-150 :, :]

    return im, bl, l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12

def remove_redundant_lead(im, gap=40):
    bl = np.argmax(histogram(im, 0))
    im_ = 255*np.ones_like(im)
    for r in range(3):
        pt, cnt = 0, 0
        while True:
            x = np.random.randint(im.shape[1])
            if im[bl,x] == 0:
                pt = x
                break
            cnt += 1
            if cnt == 10000:
                break
            
        col_ = np.where(im[:, pt]==0)[0]
        pos= []
        for i in range(len(col_)):
            if col_[i] < bl-15 or col_[i] > bl+15:
                pos.append(i)
        col_ = np.delete(col_, pos)
        im_[col_, pt] = 0

        amax, amin = max(col_), min(col_)
        for i in range(pt, im.shape[1]):
            if amin - gap > 0 and amax + gap < im.shape[0]:
                mask = np.where(im[amin-gap:amax+gap, i]==0)[0] + amin-gap
            elif amin - gap < 0 and amax + gap < im.shape[0]:
                mask = np.where(im[0:amax+gap, i]==0)[0]
            elif amin - gap < 0 and amax + gap > im.shape[0]:
                mask = np.where(im[amin-gap:im.shape[0], i]==0)[0] + amin-gap
            else:
                mask = np.where(im[0:im.shape[0], i]==0)[0]
            im_[mask, i] = 0
            try:
                amax, amin = max(mask), min(mask)
            except:
                pass

        amax, amin = max(col_), min(col_)
        for i in range(pt-1, -1, -1):
            if amin - gap > 0 and amax + gap < im.shape[0]:
                mask = np.where(im[amin-gap:amax+gap, i]==0)[0] + amin-gap
            elif amin - gap < 0 and amax + gap < im.shape[0]:
                mask = np.where(im[0:amax+gap, i]==0)[0]
            elif amin - gap > 0 and amax + gap > im.shape[0]:
                mask = np.where(im[amin-gap:im.shape[0], i]==0)[0] + amin-gap
            else:
                mask = np.where(im[0:im.shape[0], i]==0)[0]
    
            im_[mask, i] = 0
            try:
                amax, amin = max(mask), min(mask)
            except:
                pass
    return im_ 

def digitization(im, w=16, h=80, fs=200):
    im = im[:, 20:im.shape[1]-20]
    hist = histogram(im, 0)
    bl = np.argmax(hist)
    
    sig = []
    for i in range(im.shape[1]):
        col = im[:, i]
        if len(np.where(col==0)[0]) != 0:
            mag = np.mean(np.where(col==0)[0])
        else:
            mag = bl
        sig.append(mag)
    sig = (np.array(sig) - bl)/h + 1

    n_samples = int(fs*im.shape[1]/w)
    sig = resample(sig, n_samples)
    return 1-sig


def ecg(img_path):
    img = read_img(img_path)
    bl = find_baseline(img)
    _, (w, h) = find_ref_pulse(img, bl[0])
    
    img, bl, l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12 = detact_leads(img, bl)
        
    ll0 = remove_redundant_lead(l0, gap=30); ll0 = cv2.medianBlur(ll0, 3); sig0 = digitization(ll0, w, h)
    ll1 = remove_redundant_lead(l1, gap=30); ll1 = cv2.medianBlur(ll1, 3); sig1 = digitization(ll1, w, h)
    ll2 = remove_redundant_lead(l2, gap=30); ll2 = cv2.medianBlur(ll2, 3); sig2 = digitization(ll2, w, h)
    ll3 = remove_redundant_lead(l3, gap=30); ll3 = cv2.medianBlur(ll3, 3); sig3 = digitization(ll3, w, h)
    ll4 = remove_redundant_lead(l4, gap=30); ll4 = cv2.medianBlur(ll4, 3); sig4 = digitization(ll4, w, h)
    ll5 = remove_redundant_lead(l5, gap=30); ll5 = cv2.medianBlur(ll5, 3); sig5 = digitization(ll5, w, h)
    ll6 = remove_redundant_lead(l6, gap=30); ll6 = cv2.medianBlur(ll6, 3); sig6 = digitization(ll6, w, h)
    ll7 = remove_redundant_lead(l7, gap=30); ll7 = cv2.medianBlur(ll7, 3); sig7 = digitization(ll7, w, h)
    ll8 = remove_redundant_lead(l8, gap=30); ll8 = cv2.medianBlur(ll8, 3); sig8 = digitization(ll8, w, h)
    ll9 = remove_redundant_lead(l9, gap=30); ll9 = cv2.medianBlur(ll9, 3); sig9 = digitization(ll9, w, h)
    ll10 = remove_redundant_lead(l10, gap=30); ll10 = cv2.medianBlur(ll10, 3); sig10 = digitization(ll10, w, h)
    ll11 = remove_redundant_lead(l11, gap=30); ll11 = cv2.medianBlur(ll11, 3); sig11 = digitization(ll11, w, h)
    ll12 = remove_redundant_lead(l12, gap=30); ll12 = cv2.medianBlur(ll12, 3); sig12 = digitization(ll12, w, h)

    N = len(sig0)
    sig = np.zeros((12, N))
    sig[0] = sig0; sig[1] = sig1; sig[2] = sig2; sig[3] = sig3; sig[4] = sig4; sig[5] = sig5; 
    sig[6] = sig6; sig[7] = sig7; sig[8] = sig8; sig[9] = sig9; sig[10] = sig10; sig[11] = sig11
    
    # flatten sig to 1-dim python list
    res = sig.flatten()
    res = res.tolist()
    
    #return res
    result = dict()
    result['I'] = sig[0].tolist(); result['II'] = sig[1].tolist(); result['III'] = sig[2].tolist()
    result['aVR'] = sig[3].tolist(); result['aVL'] = sig[4].tolist(); result['aVF'] = sig[5].tolist()
    result['V1'] = sig[6].tolist(); result['V2'] = sig[7].tolist(); result['V3'] = sig[8].tolist()
    result['V4'] = sig[9].tolist(); result['V5'] = sig[10].tolist(); result['V6'] = sig[11].tolist()
    
    return result

if __name__ == "__main__":
    res = ecg('./ecg.jpg')
    print(res.shape)


