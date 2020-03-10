from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import sys
import argparse
import glob
import  cv2
from scipy import misc

sys.path.insert(0, './recognition/src')
import facenet_ext

class Predict_cap(object):
    def __init__(self, args):
        self.args = args

        print('Loading facenet models....')
        with tf.device('/cpu:0'):
            ########### load face verif_expression model #############################
            # Load the model of face verification
            print('Model directory: %s' % args.model_dir)
            meta_file, ckpt_file = facenet_ext.get_model_filenames(os.path.expanduser(args.model_dir))

            # facenet.load_model(args.model_dir, meta_file, ckpt_file)
            self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
            model_dir_exp = os.path.expanduser(args.model_dir)
            saver = tf.train.import_meta_graph(os.path.join(args.model_dir, meta_file))
            saver.restore(self.sess, os.path.join(model_dir_exp, ckpt_file))

            self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            self.keep_probability_placeholder = tf.get_default_graph().get_tensor_by_name('keep_probability:0')
            self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name('phase_train:0')
            self.logits = tf.get_default_graph().get_tensor_by_name('logits:0')
            self.phase_train_placeholder_expression = tf.get_default_graph().get_tensor_by_name(
                'phase_train_expression:0')

    def recog_cap(self, image):
        image_size = self.args.image_size

        #img = cv2.resize(images,(image_size,image_size))

        #images = facenet_ext.load_data(images_path, False, False, 160)
        images = facenet_ext.load_data_im(image, False, False, 160)
        if len(images.shape) == 3:
            images = np.expand_dims(images, axis=0)

        feed_dict = {self.phase_train_placeholder: False, self.phase_train_placeholder_expression: False,
                     self.images_placeholder: images, self.keep_probability_placeholder: 1.0}

        logits_array = self.sess.run([self.logits], feed_dict=feed_dict)

        logits0 = logits_array[0]
        cap_probs = np.exp(logits0) / np.sum(np.exp(logits0))
        iscap = np.argmax(logits0)
        cap_prob = cap_probs[0][iscap]




        return iscap, cap_prob


def main(args):
    predict_cap = Predict_cap(args)

    # images = glob.glob(args.images + '/*.png')
    # images.sort()

    image_list_test, label_list_test, usage_list_test, nrof_classes_test \
        = facenet_ext.get_image_paths_and_labels_headcap(
        args.images,
        'Test',
        args.nfold,
        args.ifold)

    iscap_list = []
    image_test = image_list_test[0:]
    label_test = label_list_test[0:]
    i = 0
    for image in image_test[0:]:
        #img_cv = cv2.imread(image) #### !!!!!! color image loaded by opencv is BGR
        img_misc = misc.imread(image) #### !!!!! color image loaded by sci or matplotlib (python lib) is RGB, and tensorflow lib is RGB
        ##### Pay attention, the recognition model is based on RGB image but not BGR loaded by opencv ###############
        iscap, cap_prob = predict_cap.recog_cap(img_misc)
        # if iscap == 0:
        #     print('%s NO CAP prob : %f'%(image, cap_prob))
        # else:
        #     print('%s WEARING CAP prob : %f'%(image, cap_prob))
        if cap_prob == 0.5:
            print('%d : %s'%(i, image))
            i += 1
        iscap_list.append(iscap)
    print ('cap_prob=0.5 %d'%i)


    label_test_array = np.array(label_test)
    iscap_list_array = np.array(iscap_list)
    tp = np.sum(np.logical_and(iscap_list_array, label_test_array))
    tn = np.sum(np.logical_and(np.logical_not(iscap_list_array), np.logical_not(label_test_array)))

    acc_eval = (tp+tn)/len(label_test)



    print('Prediction acc is %f'%acc_eval)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--images', type=str, help='Directory with unaligned image 1.',
                        default='/mnt/hgfs/share/data/manji/2000_labeled_sample_head_180')
    parser.add_argument('--model_dir', type=str,
                        help='Directory containing the metagraph (.meta) file and the checkpoint (ckpt) file containing model parameters',
                        default='/mnt/hgfs/share/models/20190122-191327_model/best_model')  # ../model/20170501-153641#20161217-135827#20170131-234652
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--nfold', type=int,
                        help='The ith fold used in the n-fold cross-validation', default=10)
    parser.add_argument('--ifold', type=int,
                        help='The ith fold used in the n-fold cross-validation', default=4)

    return parser.parse_args(argv)




if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
