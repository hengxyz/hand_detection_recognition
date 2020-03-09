"""Functions for building the face recognition network.
"""
# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from scipy import misc
import matplotlib.pyplot as plt
import sklearn
if sklearn.__version__ < '0.20':
    from sklearn.cross_validation import KFold ## < sklearn 0.20
else:
    from sklearn.model_selection import KFold ## > sklearn 0.20
from scipy import interpolate
import random
import re
from collections import Counter
import matplotlib.pyplot as plt
import cv2
import python_getdents
from scipy import spatial
import sys
import numpy as np
import pandas

from scipy import misc

#### libs of DavaideSanderburg ####
sys.path.insert(0, '../lib/facenet/src')
#import facenet
import glob


def label_mapping(label_list_src, EXPRSSIONS_TYPE_src, EXPRSSIONS_TYPE_trg):
    labels_mapping = []
    idx_label_notexist = []
    for i, label in enumerate(label_list_src):
        expre_src = str.split(EXPRSSIONS_TYPE_src[label], '=')[1]
        expre_trg = [x for x in EXPRSSIONS_TYPE_trg if expre_src in x]
        if expre_trg == []:
            label_trg = -1
            idx_label_notexist.append(i)
        else:
            label_trg = int(str.split(expre_trg[0], '=')[0])
        labels_mapping.append(label_trg)

    return idx_label_notexist, labels_mapping


def gather(data, label):
    i = 0
    if data.ndim == 1:
        data_batch = np.zeros(len(label))
        for idx in label:
            data_batch[i] = data[idx]
            i += 1
    if data.ndim == 2:
        data_batch = np.zeros([len(label), np.shape(data)[1]])
        for idx in label:
            data_batch[i, :] = data[idx, :]
            i += 1
    if data.ndim > 2:
        print('The data of dimension should be less than 3!\n')
        assert (data.ndim < 3)

    return data_batch


# def scatter(data, index):
#     return data_sactter

def generate_labels_id(subs):
    subjects = list(set(subs))
    subjects = np.sort(subjects)
    labels_id = []
    for sub in subs:
        labels_id.append([idx for idx, subject in enumerate(subjects) if sub == subject][0])

    return labels_id

def get_image_paths_and_labels_hand(images_path, labelfile, nfold, ifold):

    image_paths = []
    labels = []
    idx_train_all = []
    idx_test_all = []
    image_paths_final = []
    labels_final = []
    image_paths_final_test = []
    labels_final_test = []

    datal = pandas.read_excel(labelfile)
    labels_all = datal['PersonID'].values
    labels_frm = datal['Frame'].values
    labels_frm_list = labels_frm.tolist()
    labels_all_list = labels_all.tolist()

    image_paths = glob.glob(os.path.join(images_path, '*.png'))
    image_paths.sort()
    for imgfile in image_paths:
        strtmp = str.split(imgfile,'/')[-1]
        strtmp = str.split(strtmp, '_')[0]
        framenum = int(strtmp[5:])

        idx = labels_frm_list.index(framenum)
        labels.append(labels_all_list[idx])


    # folds = KFold(n=len(labels_flat), n_folds=nrof_folds, shuffle=True)
    if sklearn.__version__ < '0.20':
        folds = KFold(n=len(labels), n_folds=10, shuffle=True) ## Before the version of sklearn 0.20
    else:
        kf = KFold(n_splits=nfold, shuffle=True) ## After the version of sklearn 0.20

    i = 0

    if sklearn.__version__ < '0.20':
        for idx_train, idx_test in folds: ## Before sklearn 0.20
            idx_train_all.append([])
            idx_train_all[i].append(idx_train)
            idx_test_all.append([])
            idx_test_all[i].append(idx_test)
            # print('train:', idx_train, 'test', idx_test)
            i += 1
    else:
        for idx_train, idx_test in kf.split(labels):  ## After skleran 0.20
            idx_train_all.append([])
            idx_train_all[i].append(idx_train)
            idx_test_all.append([])
            idx_test_all[i].append(idx_test)
            #print('train:', idx_train, 'test', idx_test)
            i += 1

    idx_train = idx_train_all[ifold][0]
    idx_test = idx_test_all[ifold][0]


    for idx in idx_train:
        #idx_train.append(idx)
        image_paths_final.append(image_paths[idx])
        labels_final.append(labels[idx])


    for idx in idx_test:
        #idx_test.append(idx)
        image_paths_final_test.append(image_paths[idx])
        labels_final_test.append(labels[idx])

    nrof_classes = len(set(labels_final))
    nrof_classes_test = len(set(labels_final_test))

    return image_paths_final, labels_final, nrof_classes, image_paths_final_test, labels_final_test, nrof_classes_test

def get_image_paths_and_labels_headcap(images_path, usage, nfold, ifold):

    image_paths = []
    labels = []
    idx_train_all = []
    idx_test_all = []
    image_paths_final = []
    labels_final = []


    folders = os.listdir(images_path)
    folders.sort()
    for fold in folders:
        if not os.path.isdir(os.path.join(images_path, fold)):
            continue
        img_path_folder = glob.glob(os.path.join(images_path, fold, '*.png'))
        img_path_folder.sort()
        image_paths += img_path_folder
        label_txt = glob.glob(os.path.join(images_path, fold, '*.txt'))[0]
        with open(label_txt, 'r') as f:
            for line in f.readlines():
                line = line.replace('\r\n','\n')
                #print ('%s   %s'%(fold, line))
                labels.append(int(line[-2:-1]))

    # folds = KFold(n=len(labels_flat), n_folds=nrof_folds, shuffle=True)
    if sklearn.__version__ < '0.20':
        folds = KFold(n=len(labels), n_folds=10, shuffle=False) ## Before the version of sklearn 0.20
    else:
        kf = KFold(n_splits=nfold, shuffle=False) ## After the version of sklearn 0.20

    i = 0

    if sklearn.__version__ < '0.20':
        for idx_train, idx_test in folds: ## Before sklearn 0.20
            idx_train_all.append([])
            idx_train_all[i].append(idx_train)
            idx_test_all.append([])
            idx_test_all[i].append(idx_test)
            # print('train:', idx_train, 'test', idx_test)
            i += 1
    else:
        for idx_train, idx_test in kf.split(labels):  ## After skleran 0.20
            idx_train_all.append([])
            idx_train_all[i].append(idx_train)
            idx_test_all.append([])
            idx_test_all[i].append(idx_test)
            #print('train:', idx_train, 'test', idx_test)
            i += 1

    idx_train = idx_train_all[ifold][0]
    idx_test = idx_test_all[ifold][0]

    if usage == 'Training':
        for idx in idx_train:
            #idx_train.append(idx)
            image_paths_final.append(image_paths[idx])
            labels_final.append(labels[idx])

    if usage == 'Test':
        for idx in idx_test:
            #idx_test.append(idx)
            image_paths_final.append(image_paths[idx])
            labels_final.append(labels[idx])

    nrof_classes = len(set(labels_final))
    return image_paths_final, labels_final, usage, nrof_classes





def get_image_paths_and_labels_recog(dataset):
    image_paths_flat = []
    labels_flat = []
    classes_flat = []
    for i in range(len(dataset)):
        image_paths_flat += dataset[i].image_paths
        classes_flat += [dataset[i].name]
        labels_flat += [i] * len(dataset[i].image_paths)

    return image_paths_flat, labels_flat, classes_flat


def random_rotate_image(image):
    # angle = np.random.uniform(low=-10.0, high=10.0)
    angle = np.random.uniform(low=-180.0, high=180.0)
    return misc.imrotate(image, angle, 'bicubic')


def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y


def crop(image, random_crop, image_size):
    if image.shape[1] > image_size:
        sz1 = np.int(image.shape[1] // 2)  ##python 3 // int division
        sz2 = np.int(image_size // 2)
        if random_crop:
            diff = sz1 - sz2
            (h, v) = (np.random.randint(-diff, diff + 1), np.random.randint(-diff, diff + 1))
        else:
            (h, v) = (0, 0)
        image = image[(sz1 - sz2 + v):(sz1 + sz2 + v), (sz1 - sz2 + h):(sz1 + sz2 + h), :]
    return image


def flip(image, random_flip):
    if random_flip and np.random.choice([True, False]):
        image = np.fliplr(image)
    return image


def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret




def load_data_test(image_paths, do_random_crop, do_random_flip, image_size, do_prewhiten=True):
    nrof_samples = len(image_paths)
    images = np.zeros((nrof_samples, image_size, image_size, 3))
    for i in range(nrof_samples):
        img = misc.imread(image_paths[i])
        img = cv2.resize(img, (image_size, image_size))
        if img.ndim == 2:
            img = to_rgb(img)
        if do_prewhiten:
            img = prewhiten(img)
        img = cv2.resize(img, (image_size, image_size))
        ##img = crop(img, do_random_crop, image_size)
        img = flip(img, do_random_flip)
        images[i, :, :, :] = img
    return images


def load_data_mega(image_paths, do_random_crop, do_random_flip, do_resize, image_size, BBox, do_prewhiten=True):
    nrof_samples = len(image_paths)
    images = np.zeros((nrof_samples, image_size, image_size, 3))
    for i in range(nrof_samples):
        image = misc.imread(image_paths[i])
        BBox = BBox.astype(int)
        img = image[BBox[i, 0]:BBox[i, 0] + BBox[i, 2], BBox[i, 1]:BBox[i, 1] + BBox[i, 3], :]
        if img.ndim == 2:
            img = to_rgb(img)
        if do_prewhiten:
            img = prewhiten(img)
        if do_resize:
            img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
        img = crop(img, do_random_crop, image_size)
        img = flip(img, do_random_flip)
        images[i, :, :, :] = img

    return images




def get_learning_rate_from_file(filename, epoch):
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.split('#', 1)[0]
            if line:
                par = line.strip().split(':')
                e = int(par[0])
                lr = float(par[1])
                if e <= epoch:
                    learning_rate = lr
                    # else:
                    #     return learning_rate

        return learning_rate


def get_dataset(paths):
    dataset = []
    for path in paths.split(':'):
        path_exp = os.path.expanduser(path)
        classes = os.listdir(path_exp)
        classes.sort()
        nrof_classes = len(classes)
        for i in range(nrof_classes):
            class_name = classes[i]
            facedir = os.path.join(path_exp, class_name)
            if os.path.isdir(facedir):
                images = os.listdir(facedir)
                image_paths = [os.path.join(facedir, img) for img in images]
                dataset.append(ImageClass(class_name, image_paths))

    return dataset


def get_huge_dataset(paths, start_n=0, end_n=-1):
    dataset = []
    classes = []
    for path in paths.split(':'):
        path_exp = os.path.expanduser(path)
        for (d_ino, d_off, d_reclen, d_type, d_name) in python_getdents.getdents64(path_exp):
            if d_name == '.' or d_name == '..':
                continue
            classes += [d_name]

        classes.sort()
        nrof_classes = len(classes)
        if end_n == -1:
            end_n = nrof_classes
        if end_n > nrof_classes:
            raise ValueError('Invalid end_n:%d more than nrof_class:%d' % (end_n, nrof_classes))
        for i in range(start_n, end_n):
            if (i % 1000 == 0):
                print('reading identities: %d/%d\n' % (i, end_n))
            class_name = classes[i]
            facedir = os.path.join(path_exp, class_name)
            if os.path.isdir(facedir):
                images = os.listdir(facedir)
                image_paths = [os.path.join(facedir, img) for img in images]
                dataset.append(ImageClass(class_name, image_paths))

    return dataset


class ImageClass():
    "Stores the paths to images for a given class"

    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)

def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files)==0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files)>1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups())>=2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file

def load_model(model_dir, meta_file, ckpt_file):
    model_dir_exp = os.path.expanduser(model_dir)
    saver = tf.train.import_meta_graph(os.path.join(model_dir_exp, meta_file))
    saver.restore(tf.get_default_session(), os.path.join(model_dir_exp, ckpt_file))



def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y

def crop(image, random_crop, image_size):
    if min(image.shape[0], image.shape[1]) > image_size:
        sz1 = image.shape[0] // 2
        sz2 = image.shape[1] // 2

        crop_size = image_size//2
        diff_h = sz1 - crop_size
        diff_v = sz2 - crop_size
        (h, v) = (np.random.randint(-diff_h, diff_h + 1), np.random.randint(-diff_v, diff_v + 1))

        image = image[(sz1+h-crop_size):(sz1+h+crop_size ), (sz2+v-crop_size):(sz2+v+crop_size ), :]
    else:
        print("Image size is small than crop image size!")

    return image


# def crop(image, random_crop, image_size):
#     ## Firstly crop the image as a square according to the y length of the input image
#     if image.shape[1] > image_size:
#         sz1 = image.shape[1] // 2
#         sz2 = image_size // 2
#         if random_crop:
#             diff = sz1 - sz2
#             (h, v) = (np.random.randint(-diff, diff + 1), np.random.randint(-diff, diff + 1))
#         else:
#             (h, v) = (0, 0)
#         image = image[(sz1 - sz2 + v):(sz1 + sz2 + v), (sz1 - sz2 + h):(sz1 + sz2 + h), :]
#     return image


def flip(image, random_flip):
    if random_flip and np.random.choice([True, False]):
        image = np.fliplr(image)
    return image


def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret

def load_data(image_paths, do_random_crop, do_random_flip, image_size, do_prewhiten=True):
    nrof_samples = len(image_paths)
    images = np.zeros((nrof_samples, image_size, image_size, 3))
    for i in range(nrof_samples):
        img = misc.imread(image_paths[i])
        if img.ndim == 2:
            img = to_rgb(img)
        if do_prewhiten:
            img = prewhiten(img)
        if do_random_crop:
            img = crop(img, do_random_crop, image_size)
        if do_random_flip:
            img = flip(img, do_random_flip)
        img = cv2.resize(img,(image_size,image_size))
        images[i,:,:,:] = img
    return images

def load_data_im(imgs, do_random_crop, do_random_flip, image_size, do_prewhiten=True):
    # nrof_samples = len(image_paths)
    if (len(imgs.shape) > 3):##RGB images
        nrof_samples = imgs.shape[0]
    elif (len(imgs.shape) == 3): ## one RGB
        nrof_samples = 1
    elif (len(imgs.shape) == 2): ## grey images
        nrof_samples = imgs.shape[0]
    elif (len(imgs.shape) == 1): ## one grey
        nrof_samples = 1
    else:
        print('No images!')
        return -1

    images = np.zeros((nrof_samples, image_size, image_size, 3))
    for i in range(nrof_samples):
        # img = misc.imread(image_paths[i])
        if len(imgs.shape) == 3 or len(imgs.shape) == 1:
            img = imgs
        else:
            img = imgs[i]

        if len(img):
            if img.ndim == 2:
                img = to_rgb(img)
            if do_prewhiten:
                img = prewhiten(img)
            if do_random_crop:
                img = crop(img, do_random_crop, image_size)
            if do_random_flip:
                img = flip(img, do_random_flip)

            img = cv2.resize(img, (image_size, image_size))
            images[i] = img
    images = np.squeeze(images)
    return images


def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    folds = KFold(n=nrof_pairs, n_folds=nrof_folds, shuffle=False)
    # folds = KFold(n=nrof_pairs, n_folds=nrof_folds, shuffle=True, seed=666)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    best_threshold = np.zeros((nrof_folds))

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)

    for fold_idx, (train_set, test_set) in enumerate(folds):

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx], fp_idx, fn_idx = calculate_accuracy(threshold, dist[train_set],
                                                                                actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        best_threshold[fold_idx] = thresholds[best_threshold_index]

        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _, fp_idx, fn_idx = calculate_accuracy(
                threshold, dist[test_set], actual_issame[test_set])
        _, _, accuracy[fold_idx], fp_idx, fn_idx = calculate_accuracy(thresholds[best_threshold_index], dist[test_set],
                                                                      actual_issame[test_set])

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    mean_best_threshold = np.mean(best_threshold)

    # #### Global evaluation (not n-fold evaluation) for collecting the indices of the False positive/negative  examples  #####
    _, _, acc_total, fp_idx, fn_idx = calculate_accuracy(mean_best_threshold, dist, actual_issame)

    return tpr, fpr, accuracy, fp_idx, fn_idx, mean_best_threshold


def calculate_roc_cosine(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    folds = KFold(n=nrof_pairs, n_folds=nrof_folds, shuffle=False)
    # folds = KFold(n=nrof_pairs, n_folds=nrof_folds, shuffle=True, seed=666)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))

    # diff = np.subtract(embeddings1, embeddings2) ###Eucldian l2 distance
    # dist = np.sum(np.square(diff), 1)

    dist_all = spatial.distance.cdist(embeddings1, embeddings2,
                                      'cosine')  ## cosine_distance = 1 - similarity; similarity=dot(u,v)/(||u||*||v||)
    dist = dist_all.diagonal()

    for fold_idx, (train_set, test_set) in enumerate(folds):

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx], fp_idx, fn_idx = calculate_accuracy(threshold, dist[train_set],
                                                                                actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _, fp_idx, fn_idx = calculate_accuracy(
                threshold,
                dist[test_set],
                actual_issame[
                    test_set])
        _, _, accuracy[fold_idx], fp_idx, fn_idx = calculate_accuracy(thresholds[best_threshold_index], dist[test_set],
                                                                      actual_issame[test_set])

        tpr = np.mean(tprs, 0)
        fpr = np.mean(fprs, 0)
        best_threshold = thresholds[best_threshold_index]

        # #### Global evaluation (not n-fold evaluation) for collecting the indices of the False positive/negative  examples  #####
        _, _, acc_total, fp_idx, fn_idx = calculate_accuracy(best_threshold, dist, actual_issame)

    return tpr, fpr, accuracy, fp_idx, fn_idx, best_threshold


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size

    # ####################################  Edit by mzh 11012017   ####################################
    # #### save the false predict samples: the false posivite (fp) or the false negative(fn) #####
    fp_idx = np.logical_and(predict_issame, np.logical_not(actual_issame))
    fn_idx = np.logical_and(np.logical_not(predict_issame), actual_issame)
    # ####################################  Edit by mzh 11012017   ####################################

    return tpr, fpr, acc, fp_idx, fn_idx


def plot_roc(fpr, tpr, label):
    figure = plt.figure()
    plt.plot(fpr, tpr, label=label)
    plt.title('Receiver Operating Characteristics')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.plot([0, 1], [0, 1], 'g--')
    plt.grid(True)
    plt.show()

    return figure


def calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)

    nrof_thresholds = len(thresholds)
    folds = KFold(n=nrof_pairs, n_folds=nrof_folds, shuffle=False)
    # folds = KFold(n=nrof_pairs, n_folds=nrof_folds, shuffle=True, seed=666)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    for fold_idx, (train_set, test_set) in enumerate(folds):

        if nrof_thresholds > 1:
            # Find the threshold that gives FAR = far_target
            far_train = np.zeros(nrof_thresholds)
            for threshold_idx, threshold in enumerate(thresholds):
                _, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_set], actual_issame[train_set])
            if np.max(far_train) >= far_target:
                f = interpolate.interp1d(far_train, thresholds, kind='slinear')
                threshold = f(far_target)
            else:
                threshold = 0.0
        else:
            threshold = thresholds[0]

        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist[test_set], actual_issame[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)

    val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist[test_set], actual_issame[test_set])

    return val_mean, val_std, far_mean, threshold


def calculate_val_cosine(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    folds = KFold(n=nrof_pairs, n_folds=nrof_folds, shuffle=False)
    # folds = KFold(n=nrof_pairs, n_folds=nrof_folds, shuffle=True, seed=666)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    # diff = np.subtract(embeddings1, embeddings2)
    # dist = np.sum(np.square(diff), 1)
    dist_all = spatial.distance.cdist(embeddings1, embeddings2,
                                      'cosine')  ## cosine_distance = 1 - similarity; similarity=dot(u,v)/(||u||*||v||)
    dist = dist_all.diagonal()

    for fold_idx, (train_set, test_set) in enumerate(folds):

        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist[test_set], actual_issame[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean, threshold


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    if n_same > 0:
        val = float(true_accept) / float(n_same)
    else:
        val = 0
    if n_diff > 0:
        far = float(false_accept) / float(n_diff)
    else:
        far = 0
    return val, far


## get the labels of  the triplet paths for calculating the center loss - mzh edit 31012017
def get_label_triplet(triplet_paths):
    classes = []
    classes_list = []
    labels_triplet = []
    for image_path in triplet_paths:
        str_items = image_path.split('/')
        classes_list.append(str_items[-2])

    classes = list(sorted(set(classes_list), key=classes_list.index))

    for item in classes_list:
        labels_triplet.append(classes.index(item))

    return labels_triplet


def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files) == 0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files) > 1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups()) >= 2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file


def class_filter(image_list, label_list, num_imgs_class):
    counter = Counter(label_list)
    label_num = counter.values()
    label_key = counter.keys()

    idx = [idx for idx, val in enumerate(label_num) if val > num_imgs_class]
    label_idx = [label_key[i] for i in idx]
    idx_list = [i for i in range(0, len(label_list)) if label_list[i] in label_idx]
    label_list_new = [label_list[i] for i in idx_list]
    image_list_new = [image_list[i] for i in idx_list]

    # plt.hist(label_num, bins = 'auto')
    return image_list_new, label_list_new


## Select the images for a epoch in which each batch includes at least two different classes and each class has more than one image
def select_batch_images(image_list, label_list, epoch, epoch_size, batch_size, num_classes_batch, num_imgs_class):
    label_epoch = []
    image_epoch = []

    counter = Counter(label_list)
    label_num = counter.values()
    label_key = counter.keys()
    nrof_examples = len(image_list)
    nrof_examples_per_epoch = epoch_size * batch_size
    j = epoch * nrof_examples_per_epoch % nrof_examples

    if j + epoch_size * batch_size > nrof_examples:
        j = random.choice(range(0, nrof_examples - epoch_size * batch_size))

    for i in range(epoch_size):
        print('In select_batch_images, batch %d selecting...\n' % (i))
        label_batch = label_list[j + i * batch_size:j + (i + 1) * batch_size]
        image_batch = image_list[j + i * batch_size:j + (i + 1) * batch_size]

        label_unique = set(label_batch)
        if (len(label_unique) < num_classes_batch or len(label_unique) > (batch_size / num_imgs_class)):
            if (num_classes_batch > (batch_size / num_imgs_class)):
                raise ValueError(
                    'The wanted minumum number of classes in a batch (%d classes) is more than the limit can be assigned (%d classes)' % (
                    num_classes_batch, num_imgs_class))
            label_batch = []
            image_batch = []
            ## re-select the image batch which includes num_classes_batch classes
            nrof_im_each_class = np.int(batch_size / num_classes_batch)
            idx = [idx for idx, val in enumerate(label_num) if val > nrof_im_each_class]
            if (len(idx) < num_classes_batch):
                raise ValueError('No enough classes to chose!')
            idx_select = random.sample(idx, num_classes_batch)
            label_key_select = [label_key[i] for i in idx_select]
            for label in label_key_select:
                start_tmp = label_list.index(label)
                idx_tmp = range(start_tmp, start_tmp + nrof_im_each_class + 1)
                label_tmp = [label_list[i] for i in idx_tmp]
                img_tmp = [image_list[i] for i in idx_tmp]
                label_batch += label_tmp
                image_batch += img_tmp

            label_batch = label_batch[0:batch_size]
            image_batch = image_batch[0:batch_size]

        label_epoch += label_batch
        image_epoch += image_batch

    return image_epoch, label_epoch


def label_mapping(label_list_src, EXPRSSIONS_TYPE_src, EXPRSSIONS_TYPE_trg):
    labels_mapping = []
    idx_label_notexist = []
    for i, label in enumerate(label_list_src):
        expre_src = str.split(EXPRSSIONS_TYPE_src[label], '=')[1]
        expre_trg = [x for x in EXPRSSIONS_TYPE_trg if expre_src in x]
        if expre_trg == []:
            label_trg = -1
            idx_label_notexist.append(i)
        else:
            label_trg = int(str.split(expre_trg[0], '=')[0])
        labels_mapping.append(label_trg)

    return idx_label_notexist, labels_mapping
