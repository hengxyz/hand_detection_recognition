import cv2
import glob
import sys
import argparse
import os

def main(args):
    image_path = args.images
    fps = 5
    video_name = str.split(image_path, '/')[-1]
    #fourcc = cv2.VideoWriter_fourcc('M','J','P','G') ##opecv 3.x
    fourcc = cv2.cv.CV_FOURCC('M','J','P','G') ### opencv 2.x
    videoWriter = cv2.VideoWriter(os.path.join(image_path, video_name+'.avi'), fourcc, fps, (1920,1080))
    images = glob.glob(image_path+'/*.png')
    images.sort()
    for i, image in enumerate(images):

        print('%d: %s'%(i, image))
        img12 = cv2.imread(image)
    #    cv2.imshow('img', img12)
    #    cv2.waitKey(1000/int(fps))
        videoWriter.write(img12)
    videoWriter.release()

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--images', type=str, help='Directory with unaligned image 1.',
                        default='/mnt/hgfs/share/data/manji/2000_labeled_sample_head_180')


    return parser.parse_args(argv)

if __name__=='__main__':
    main(parse_arguments(sys.argv[1:]))