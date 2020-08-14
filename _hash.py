import cv2
import numpy as np


def ahash(img_mat, w=16, h=16):
    """
    Average Hash
    """
    img_mat = cv2.resize(img_mat, (w, h))
    img_mat = cv2.cvtColor(img_mat, cv2.COLOR_BGR2GRAY)
    avg = np.mean(img_mat)
    ahash = ((img_mat > avg) + 0).reshape(-1).tolist()
    # ahash = []
    # for i in range(img_mat.shape[0]):
    #     for j in range(img_mat.shape[1]):
    #         if img_mat[i, j] >= avg:
    #             ahash.append(1)
    #         else:
    #             ahash.append(0)
    return ahash


def dhash(img_mat, w=17, h=16):
    """
    Difference Hash
    """
#     print(img_mat.shape)
    img_mat = cv2.resize(img_mat, (w, h))
    img_mat = cv2.cvtColor(img_mat, cv2.COLOR_BGR2GRAY)
#     print(img.shape)
    dhash = []
    for i in range(h):
        for j in range(h):
            if img_mat[i, j] > img_mat[i, j + 1]:
                dhash.append(1)
            else:
                dhash.append(0)
    return dhash


def phash(img_mat, w=32, h=32):
    """
    Perceptual Hash
    """
    img_mat = cv2.resize(img_mat, (w, h))
    img_gray = cv2.cvtColor(img_mat, cv2.COLOR_BGR2GRAY)
    dct = cv2.dct(np.float32(img_gray))  # DCT 
    dct_roi = dct[0:16, 0:16]
    dct_avg = np.mean(dct_roi)
    phash_mat = (dct_roi > dct_avg) + 0
    phash = phash_mat.reshape(-1).tolist()
    return phash


def phash_cv2(img):
    cv2_phash = cv2.img_hash.pHash
    return cv2_phash(img).tolist()[0]
    
    
def phash_hsv(img_mat, w=32, h=32):
    img_mat = cv2.resize(img_mat, (w, h))
    img_hsv = cv2.cvtColor(img_mat, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)
    index_min, index_max = 0, 12
    dct_h = cv2.dct(np.float32(h))[index_min:index_max, index_min:index_max]
    dct_s = cv2.dct(np.float32(s))[index_min:index_max, index_min:index_max]
    dct_v = cv2.dct(np.float32(v))[index_min:index_max, index_min:index_max]
    
    dct_avg_h = np.mean(dct_h)
    dct_avg_s = np.mean(dct_s)
    dct_avg_v = np.mean(dct_v)
    phash_h = ((dct_h > dct_avg_h) + 0).reshape(-1).tolist()
    phash_s = ((dct_s > dct_avg_s) + 0).reshape(-1).tolist()
    phash_v = ((dct_v > dct_avg_v) + 0).reshape(-1).tolist()
    
    phash_hsv = []
    phash_hsv.extend(phash_h)
    phash_hsv.extend(phash_s)
    phash_hsv.extend(phash_v)
    return phash_hsv


def color_moments(img_mat):
    """
    :param img_mat: image load by cv2
    :return: color moment of image
             cm: Mean, Standard Deviation and Skewness value on H, S, V channel
    """
    img_hsv = cv2.cvtColor(img_mat, cv2.COLOR_BGR2HSV)
    img_h, img_s, img_v = cv2.split(img_hsv)
    cm = []
    cm.extend([np.mean(img_h), np.mean(img_s), np.mean(img_v)])
    cm.extend([np.std(img_h), np.std(img_s), np.std(img_v)])
    img_h_skew = np.power(np.mean(abs(img_h - np.mean(img_h))), 1/3)
    img_s_skew = np.power(np.mean(abs(img_s - np.mean(img_s))), 1/3)
    img_v_skew = np.power(np.mean(abs(img_v - np.mean(img_v))), 1/3)
    cm.extend([img_h_skew, img_s_skew, img_v_skew])
    return cm


def hamming_distance(hash_1, hash_2, hsv=False, cm=False):
    """
    Calculate hamming distance of two hash values
    """
    dis = 0
    for index in range(len(hash_1)):
        if not cm:
            if hash_1[index] != hash_2[index]:
                if not hsv:
                    dis += 1
                else:
                    if index < len(hash_1) / 3:
                        dis += 0.05
                    elif index < len(hash_1) / 3 * 2:
                        dis += 1
                    else:
                        dis += 1
        else:
            if index % 3 == 0:
                dis += 2 * abs(hash_1[index] - hash_2[index])
            elif index % 3 == 1:
                dis += 0.5 * abs(hash_1[index] - hash_2[index])
            else:   
                dis += 0.5 * abs(hash_1[index] - hash_2[index])
                
    return dis
