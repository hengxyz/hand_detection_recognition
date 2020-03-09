# from __future__ import absolute_import
# from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import glob
import numpy as np
from PIL import Image


def main():
    data = '/data/zming/GH/manji/2000_labeled_sample_head'
    folders = os.listdir(data)
    folders.sort()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        with open(os.path.join(data, 'bad_images.txt'), 'w') as f_badimg:
            image_num = 0
            for fold in folders:
                if not os.path.isdir(os.path.join(data, fold)):
                    continue
                images_path = glob.glob(os.path.join(data, fold, '*.png'))
                images_path.sort()
                for image_path in images_path:
                    image_num += 1
                    #print ('\r','%s'%image_path, end = '')
                    print(image_path)
                    filename_queue = tf.train.string_input_producer([image_path]) #  list of files to read

                    reader = tf.WholeFileReader()
                    key, value = reader.read(filename_queue)

                    my_img = tf.image.decode_png(value)  # use png or jpg decoder based on your files.

                    init_op = tf.global_variables_initializer()

                    sess.run(init_op)

                    # Start populating the filename queue.

                    coord = tf.train.Coordinator()
                    threads = tf.train.start_queue_runners(coord=coord)

                    try:
                        for i in range(1): #length of your filename list
                            image = my_img.eval() #here is your image Tensor :)
                            # print(image.shape)
                            # Image.fromarray(np.asarray(image)).show()
                    except:
                        print('!!!!!!!!!!!!! %s'%image_path)
                        f_badimg.write(image_path)
                        continue


        f_badimg.close()
        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    main()