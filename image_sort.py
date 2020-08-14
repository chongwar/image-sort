import cv2
import numpy as np
import os
import sys
from _hash import ahash, dhash, phash, phash_cv2, phash_hsv, color_moments, hamming_distance

if sys.platform == 'linux':
    # img_res_path = '/home/chongwar/code/image_hash/images/images_result'
    img_res_path = '/home/chongwar/code/image_hash/images/images_database_result'
else:
    img_res_path = r'E:\Code\Python\image_hash\images\images_result'


def img_select(hash_func, img_path, img, img_list, img_dict, count=1, img_save_path='', img_format='jpg'):
    """
    Recursively select the most similar image and save it.
    :param hash_func: type of hash function
    :param img_path: path of the unsorted image
    :param img: current image to save
    :param img_list: unsorted image name list
    :param img_dict: image info(key: image name, value: hash value of this image)
    :param count: amount of saved images
    :param img_save_path: path to save sorted image
    :param img_format: image format
    :return: img_select function
    """
    if len(img_list) == 1:
        img_tmp = cv2.imread(os.path.join(img_path, img_list[0]))
        img_new_name = os.path.join(img_save_path, '{:0>3d}_{}.{}'.format(count, img.split('.')[0], img_format))
        cv2.imwrite(img_new_name, img_tmp)
        return print('Sort finish.')

    # save current img
    img_save_path = img_save_path
    img_format = img_format
    img_tmp = cv2.imread(os.path.join(img_path, img))
    img_new_name = os.path.join(img_save_path, '{:0>3d}_{}.{}'.format(count, img.split('.')[0], img_format))
    cv2.imwrite(img_new_name, img_tmp)
    # delete the current img in the list
    img_list.pop(img_list.index(img))
    # generate the hash of current img
    hash_std = hash_func(img_tmp)

    # find the most similar img
    dis_min = 10000 
    for img in img_list:
        hash_now = img_dict[img]
        dis = hamming_distance(hash_std, hash_now, 
                               hsv=True if hash_func == phash_hsv else False, 
                               cm=True if hash_func == color_moments else False)
        if dis < dis_min:
            dis_min = dis
            img_min = img
    count += 1
    return img_select(hash_func, img_path, img_min, img_list, img_dict, count, img_save_path, img_format)


def image_sort(hash_func, img_path):
    """
    Sort the image.
    :param hash_func: type of hash function
    :param img_path: path of the unsorted image
    """
    if sys.platform == 'linux':
        fold_name = img_path.split('/')[-1]
    else:
        fold_name = img_path.split('\\')[-1]

    img_format = os.listdir(img_path)[0].split('.')[1]

    if hash_func == ahash:
        if sys.platform == 'linux':
            img_save_path = '/home/chongwar/code/image_hash/images/images_result_ahash'
        else:
            img_save_path = r'E:\Code\Python\image_hash\images\images_result_ahash'
        func_name = 'ahash'
        
    elif hash_func == dhash:
        if sys.platform == 'linux':
            img_save_path = '/home/chongwar/code/image_hash/images/images_result_dhash'
        else:
            img_save_path = r'E:\Code\Python\image_hash\images\images_result_dhash'
        func_name = 'dhash'
        
    elif hash_func == phash:
        if sys.platform == 'linux':
            img_save_path = '/home/chongwar/code/image_hash/images/images_result_phash'
        else:
            img_save_path = r'E:\Code\Python\image_hash\images\images_result_phash'
        func_name = 'phash'
        
    elif hash_func == phash_cv2:
        if sys.platform == 'linux':
            img_save_path = '/home/chongwar/code/image_hash/images/images_result_phash_cv2'
        else:
            img_save_path = r'E:\Code\Python\image_hash\images\images_result_phash_cv2'
        func_name = 'phash_cv2' 
        
    elif hash_func == phash_hsv:
        if sys.platform == 'linux':
            img_save_path = '/home/chongwar/code/image_hash/images/images_result_phash_hsv'
        else:
            img_save_path = r'E:\Code\Python\image_hash\images\images_result_phash_hsv'
        func_name = 'phash_hsv'
        
    elif hash_func == color_moments:
        if sys.platform == 'linux':
            img_save_path = '/home/chongwar/code/image_hash/images/images_result_color_moments'
        else:
            img_save_path = r'E:\Code\Python\image_hash\images\images_result_color_moments'
        func_name = 'color_moments'
        
    else:
        raise ValueError('No such hash function.')
    
    img_save_path = os.path.join(img_save_path, fold_name)
    if not os.path.exists(img_save_path):
        if sys.platform == 'linux':
            os.makedirs(img_save_path)
        else:
            os.mkdir(img_save_path)
    else:
        for i in os.listdir(img_save_path):
            # delete the former imgs
            os.remove(os.path.join(img_save_path, i))
    
    img_list = os.listdir(img_path)
    img_dict = {}
    for img in img_list:
        img_mat = cv2.imread(os.path.join(img_path, img))
        # save hash value in dict, the key is img name
        img_dict[img] = hash_func(img_mat)  
    
    if not os.path.exists('result_hash'):
        os.mkdir('result_hash')

    # save the original hash value in file
    # np.save('result_hash/{}_{}_dict.npy'.format(fold_name, func_name), img_dict)
    
    img_select(hash_func, img_path, img_list[0], img_list, img_dict, img_save_path=img_save_path, img_format=img_format)


def main(img_res_path=img_res_path):
    for img_folder_name in os.listdir(img_res_path):
        img_path = os.path.join(img_res_path, img_folder_name)
        print('Dealing with {}...'.format(img_path.split('/' if sys.platform == 'linux' else '\\')[-1]))
        
        # image_sort(ahash, img_path)
        # image_sort(dhash, img_path)
        # image_sort(phash, img_path)
        # image_sort(phash_cv2, img_path)
        # image_sort(phash_hsv, img_path)
        image_sort(color_moments, img_path)


if __name__ == '__main__':
    main()
    