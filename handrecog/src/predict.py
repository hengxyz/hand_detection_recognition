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

#sys.path.insert(0, './recognition/src')
import facenet_ext

class Predict_hand(object):
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

    def recog_hand(self, imgs):
        #image_size = self.args.image_size

        #img = cv2.resize(images,(image_size,image_size))

        #images = facenet_ext.load_data(images_path, False, False, 160)
        images = facenet_ext.load_data_im(imgs, False, False, 160)
        if len(images.shape) == 3:
            images = np.expand_dims(images, axis=0)

        feed_dict = {self.phase_train_placeholder: False, self.phase_train_placeholder_expression: False,
                     self.images_placeholder: images, self.keep_probability_placeholder: 1.0}

        logits_ = self.sess.run([self.logits], feed_dict=feed_dict)

        # logits0 = logits_array[0]
        # hand_probs = np.exp(logits0) / np.sum(np.exp(logits0))
        # IDs = np.argmax(logits0)
        # probs = hand_probs[0][IDs]
        logits_array = np.array(logits_)
        logits_array = np.squeeze(logits_array, 0)
        exp_logit = np.exp(logits_array)
        imgs_sf_denominator = np.sum(exp_logit, 1)
        imgs_sf = exp_logit/imgs_sf_denominator
        IDs = np.argmax(imgs_sf, 1)
        probs = np.max(imgs_sf, 1)

        return IDs, probs


def main(args):
    predict_hand = Predict_hand(args)

    # images = glob.glob(args.images + '/*.png')
    # images.sort()

    image_list_train, label_list_train, nrof_classes_train, image_list_test, label_list_test, nrof_classes_test \
        = facenet_ext.get_image_paths_and_labels_hand(args.images, args.labels, args.nfold, args.ifold)

    # image_list_test, label_list_test, usage_list_test, nrof_classes_test \
    #     = facenet_ext.get_image_paths_and_labels_hand(
    #     args.images,
    #     args.nfold,
    #     args.ifold)

    ishand_list = []
    image_test = image_list_test[0:]
    label_test = label_list_test[0:]
    i = 0
    for image in image_test[0:]:
        # img_cv = cv2.imread(image) #### !!!!!! color image loaded by opencv is BGR
        img_misc = misc.imread(image)  #### !!!!! color image loaded by sci or matplotlib (python lib) is RGB, and tensorflow lib is RGB
        ##### Pay attention, the recognition model is based on RGB image but not BGR loaded by opencv ###############
        ishand, hand_prob = predict_hand.recog_hand(img_misc)
        # if ishand == 0:
        #     print('%s NO CAP prob : %f'%(image, hand_prob))
        # else:
        #     print('%s WEARING CAP prob : %f'%(image, hand_prob))
        if hand_prob == 0.5:
            print('%d : %s'%(i, image))
            i += 1
        ishand_list.append(ishand)
    print ('hand_prob=0.5 %d'%i)

    label_test_array = np.array(label_test)
    ishand_list_array = np.array(ishand_list)
    tp = np.sum(np.logical_and(ishand_list_array, label_test_array))
    tn = np.sum(np.logical_and(np.logical_not(ishand_list_array), np.logical_not(label_test_array)))

    acc_eval = (tp+tn)/len(label_test)



    print('Prediction acc is %f'%acc_eval)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--images', type=str, help='Directory with unaligned image 1.',
                        default='/data/zming/datasets/Hand/hand_frames')
    parser.add_argument('--labels', type=str,
                        help='Path to the Emotion labels file.', default='/data/zming/datasets/Hand/HandLabel_zm.xlsx')
    parser.add_argument('--model_dir', type=str,
                        help='Directory containing the metagraph (.meta) file and the checkpoint (ckpt) file containing model parameters',
                        default='/data/zming/models/hand/20190220-173420/')  # ../model/20170501-153641#20161217-135827#20170131-234652
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--nfold', type=int,
                        help='The ith fold used in the n-fold cross-validation', default=10)
    parser.add_argument('--ifold', type=int,
                        help='The ith fold used in the n-fold cross-validation', default=0)

    return parser.parse_args(argv)




if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
