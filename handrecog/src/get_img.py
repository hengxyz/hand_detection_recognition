
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import glob
import argparse
import cv2



class new_args:
    def __init__(self):
        self.images_path = []
        self.output_dir = []



def get_imgs_dirs(args):
    folders = os.listdir(args.images_path)
    folders.sort()
    args_temp = new_args()
    for folder in folders:
        images_path_folder = os.path.join(args.images_path, folder)
        output_folder = os.path.join(args.output_dir, folder)
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)

        args_temp.images_path = images_path_folder
        args_temp.output_dir = output_folder

        get_imgs(args_temp)


def get_imgs(args):
    images = glob.glob(args.images_path + '/*.png')
    images.sort()
    labels = glob.glob(args.images_path + '/*.txt')
    labels.sort()
    imgnum = 0
    with open(os.path.join(args.output_dir, 'labels.txt'), 'w') as f_cap:
        for image_path,label_path in zip(images, labels):

            image = cv2.imread(image_path)

            with open(label_path, 'r') as f:
                head_num = 0
                for line in f.readlines():
                    [xmin, ymin, x_len, y_len, label] = str.split(line, ' ')
                    ymin = int(float(ymin))
                    ymax = int(float(ymin) + float(y_len))
                    xmin = int(float(xmin))
                    xmax = int(float(xmin) + float(x_len))

                    xmin = max(0, xmin)
                    ymin = max(0, ymin)
                    xmax = min(xmax, image.shape[1])
                    ymax = min(ymax, image.shape[0])

                    label = int(float(label[:1]))
                    head_img = image[ymin:ymax, xmin:xmax]
                    cv2.imwrite(os.path.join(args.output_dir, image_path[-8:-4]+'_head_%d'%head_num+image_path[-4:]), head_img)
                    f_cap.write('%s_head_%d%s %d\n'%(image_path[-8:-4], head_num, image_path[-4:],label))
                    print('%d %s_head_%d %d\n' % (imgnum, image_path[-8:], head_num, label))
                    head_num += 1
            imgnum +=1

    f_cap.close()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--images_path', type=str,
                        help='Directory of the input images.', default='/data/zming/GH/manji/2000_labeled_sample/')
    parser.add_argument('--output_dir', type=str,
                        help='Directory of the crop head images.', default='/data/zming/GH/manji/2000_labeled_sample_head')

    return  parser.parse_args(argv)



if __name__ == '__main__':
    #get_imgs(parse_arguments(sys.argv[1:])) ## for one folder
    get_imgs_dirs(parse_arguments(sys.argv[1:])) ## for one folder
