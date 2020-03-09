# from __future__ import absolute_import
# from __future__ import division
from __future__ import print_function

import os
import glob
import numpy as np
from PIL import Image
import cv2


def main():
    data = '/data/zming/GH/manji/2000_labeled_sample_head'
    output = '/data/zming/GH/manji/2000_labeled_sample_head_180'
    size_dst = 180

    folders = os.listdir(data)
    folders.sort()

    with open(os.path.join(data, 'bad_images_cv2.txt'), 'w') as f_badimg:
        image_num = 0
        for fold in folders:
            if not os.path.isdir(os.path.join(data, fold)):
                continue
            if not os.path.isdir(os.path.join(output, fold)):
                os.mkdir(os.path.join(output, fold))
            images_path = glob.glob(os.path.join(data, fold, '*.png'))
            images_path.sort()
            for image_path in images_path:
                image_num += 1
                #print ('\r','%s'%image_path, end = '')
                print(image_path)
                img_name = str.split(image_path, '/')[-1]

                try:
                    im = cv2.imread(image_path)
                    im_resize = cv2.resize(im, (size_dst, size_dst))
                    cv2.imwrite(os.path.join(output, fold, img_name), im_resize)
                except:
                    print('!!!!!!!!!!!!! %s'%image_path)
                    f_badimg.write(image_path)
                    continue


if __name__ == '__main__':
    main()