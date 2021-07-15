import glob
import os
import cv2
import numpy as np
from tqdm import tqdm
from helper import var_info
import matplotlib.pyplot as plt

label_dict = {'angry': 0, 'disgusted': 1, 'fearful': 2, 'happy': 3, 'neutral': 4, 'sad': 5,
              'surprised': 6}

label_dict_inv = {v: k for k, v in label_dict.items()}

new_parent_dir = 'soxed_train/'
old_parent_dir = 'train'
save_dir = "./mfcc"
folds = sub_dirs = np.array(['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised'])
parent_dir = 'train'
file_ext = "*.png"


def extract_image(parent_dir, sub_dirs, max_file=400):
    feature = []
    label = []
    for sub_dir in sub_dirs:
        for fn in tqdm(glob.glob(os.path.join(parent_dir, sub_dir, file_ext))[:max_file]):
            fn = fn.replace('\\', '/')
            img_cv = cv2.imread(fn)[:, :, 0]  # 读取数据
            img_cv_edge = cv2.Canny(img_cv, 300, 170)
            feature.append([img_cv])
            label_name = sub_dir
            label.extend([label_dict[label_name]])
    return [feature, label]


def extract_image_edge(parent_dir, sub_dirs, max_file=400):
    feature = []
    label = []
    for sub_dir in sub_dirs:
        for fn in tqdm(glob.glob(os.path.join(parent_dir, sub_dir, file_ext))[:max_file]):
            fn = fn.replace('\\', '/')
            img_cv = cv2.imread(fn)[:, :, 0]  # 读取数据
            img_cv_edge = cv2.Canny(img_cv, 300, 170)
            feature.append([img_cv_edge])
            label_name = sub_dir
            label.extend([label_dict[label_name]])
    return [feature, label]


fea = extract_image(parent_dir, sub_dirs)
np.save('./feature_raw', fea[0])
np.save('./label', fea[1])

fea_edge = extract_image_edge(parent_dir, sub_dirs)
np.save('./feature_edge', fea_edge[0])
np.save('./label_edge', fea_edge[1])
