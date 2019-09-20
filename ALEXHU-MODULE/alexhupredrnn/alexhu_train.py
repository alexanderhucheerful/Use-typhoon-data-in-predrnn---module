__author__ = 'yunbo'

import os.path
import time
import numpy as np
import tensorflow as tf
import cv2
import sys
import random
from nets import models_factory
from data_provider import datasets_fuck
from utils import preprocess
from utils import metrics
from skimage.measure import compare_ssim
import matplotlib
matplotlib.use("Pdf")
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
FLAGS = tf.app.flags.FLAGS

# data I/O
tf.app.flags.DEFINE_string('dataset_name', 'fuck',
                           'The name of dataset.')
tf.app.flags.DEFINE_string('train_data_paths',
                           '/media/workdir/hujh/AlexHu-predrnn/trainpng/',
                           'train data paths.')
tf.app.flags.DEFINE_string('valid_data_paths',
                           '/media/workdir/hujh/AlexHu-predrnn/round1_testsetpng/',
                           'validation data paths.')
tf.app.flags.DEFINE_string('save_dir', '/media/workdir/hujh/AlexHu-predrnn/checkpointnew2',
                            'dir to store trained net.')
tf.app.flags.DEFINE_string('gen_frm_dir', 'D:/predtrain/result',
                           'dir to store result.')
# model
tf.app.flags.DEFINE_string('model_name', 'predrnn_pp',
                           'The name of the architecture.')
tf.app.flags.DEFINE_string('pretrained_model', '',
                           'file of a pretrained model to initialize from.')
tf.app.flags.DEFINE_integer('input_length', 6,
                            'encoder hidden states.')
tf.app.flags.DEFINE_integer('seq_length', 12,
                            'total input and output length.')
tf.app.flags.DEFINE_integer('img_width', 20,
                            'input image width.')
tf.app.flags.DEFINE_integer('img_channel', 3,
                            'number of image channel.')
tf.app.flags.DEFINE_integer('stride', 1,
                            'stride of a convlstm layer.')
tf.app.flags.DEFINE_integer('filter_size', 5,
                            'filter of a convlstm layer.')
tf.app.flags.DEFINE_string('num_hidden', '128,64,64,64',
                           'COMMA separated number of units in a convlstm layer.')
tf.app.flags.DEFINE_integer('patch_size', 1,
                            'patch size on one dimension.')
tf.app.flags.DEFINE_boolean('layer_norm', True,
                            'whether to apply tensor layer norm.')
# optimization
tf.app.flags.DEFINE_float('lr', 0.001,
                          'base learning rate.')
tf.app.flags.DEFINE_boolean('reverse_input', True,
                            'whether to reverse the input frames while training.')
tf.app.flags.DEFINE_integer('batch_size', 1,
                            'batch size for training.')
tf.app.flags.DEFINE_integer('max_iterations', 200,
                            'max num of steps.')
tf.app.flags.DEFINE_integer('display_interval', 1,
                            'number of iters showing training loss.')
tf.app.flags.DEFINE_integer('test_interval', 10,
                            'number of iters for test.')
tf.app.flags.DEFINE_integer('snapshot_interval', 10,
                            'number of iters saving models.')

class Model(object):
    def __init__(self):
        # inputs
        self.x = tf.placeholder(tf.float32,
                                [FLAGS.batch_size,
                                 FLAGS.seq_length,
                                 FLAGS.img_width//FLAGS.patch_size,
                                 FLAGS.img_width//FLAGS.patch_size,
                                 int(FLAGS.patch_size*FLAGS.patch_size*FLAGS.img_channel)])

        self.mask_true = tf.placeholder(tf.float32,
                                        [FLAGS.batch_size,
                                         FLAGS.seq_length-FLAGS.input_length-1,
                                         FLAGS.img_width//FLAGS.patch_size,
                                         FLAGS.img_width//FLAGS.patch_size,
                                         int(FLAGS.patch_size*FLAGS.patch_size*FLAGS.img_channel)])

        grads = []
        loss_train = []
        self.pred_seq = []
        self.tf_lr = tf.placeholder(tf.float32, shape=[])
        num_hidden = [int(x) for x in FLAGS.num_hidden.split(',')]
        print(num_hidden)
        num_layers = len(num_hidden)
        print(num_layers)
        with tf.variable_scope(tf.get_variable_scope()):
            # define a model
            output_list = models_factory.construct_model(
                FLAGS.model_name, self.x,
                self.mask_true,
                num_layers, num_hidden,
                FLAGS.filter_size, FLAGS.stride,
                FLAGS.seq_length, FLAGS.input_length,
                FLAGS.layer_norm)
            gen_ims = output_list[0]
            loss = output_list[1]
            pred_ims = gen_ims[:,FLAGS.input_length-1:]
            self.loss_train = loss / FLAGS.batch_size
            # gradients
            all_params = tf.trainable_variables()
            grads.append(tf.gradients(loss, all_params))
            self.pred_seq.append(pred_ims)

        self.train_op = tf.train.AdamOptimizer(FLAGS.lr).minimize(loss)

        # session
        variables = tf.global_variables()
        self.saver = tf.train.Saver(variables)
        init = tf.global_variables_initializer()
        configProt = tf.ConfigProto()
        configProt.gpu_options.allow_growth = True
        configProt.allow_soft_placement = True
        self.sess = tf.Session(config = configProt)
        self.sess.run(init)
        if FLAGS.pretrained_model:
            self.saver.restore(self.sess, FLAGS.pretrained_model)

    def train(self, inputs, lr, mask_true):
        feed_dict = {self.x: inputs}
        feed_dict.update({self.tf_lr: lr})
        feed_dict.update({self.mask_true: mask_true})
        loss, _ = self.sess.run((self.loss_train, self.train_op), feed_dict)
        return loss

    def test(self, inputs, mask_true):
        feed_dict = {self.x: inputs}
        feed_dict.update({self.mask_true: mask_true})
        gen_ims = self.sess.run(self.pred_seq, feed_dict)
        return gen_ims

    def save(self, itr):
        checkpoint_path = os.path.join(FLAGS.save_dir, 'model.ckpt')
        self.saver.save(self.sess, checkpoint_path, global_step=itr)
        print('saved to ' + FLAGS.save_dir)

def main(argv=None):
    if tf.gfile.Exists(FLAGS.save_dir):
        tf.gfile.DeleteRecursively(FLAGS.save_dir)
    tf.gfile.MakeDirs(FLAGS.save_dir)
    if tf.gfile.Exists(FLAGS.gen_frm_dir):
        tf.gfile.DeleteRecursively(FLAGS.gen_frm_dir)
    tf.gfile.MakeDirs(FLAGS.gen_frm_dir)

    # load data
    train_input_handle, test_input_handle = datasets_fuck.data_provider(
        FLAGS.dataset_name, FLAGS.train_data_paths, FLAGS.valid_data_paths,
        FLAGS.batch_size, FLAGS.img_width,FLAGS.img_channel)
    print("the input_handle_state is ")
   # print(train_input_handle)
   

    print("Initializing models")
    model = Model()
    lr = FLAGS.lr

    delta = 0.00002
    base = 0.99998
    eta = 1
    # this x ,y is ready for save data of loss to show direct
    y = []
    x = []

    for itr in range(1, FLAGS.max_iterations + 1):
        if train_input_handle.no_batch_left():
            train_input_handle.begin()
        else:
        	  train_input_handle.initial()
        ims = train_input_handle.get_batch()
        #$print(ims)
        #ims = preprocess.reshape_patch(ims, FLAGS.patch_size)
   
     #print("fuck you ims shape")
        print(ims.shape)
        if itr < 50000:
            eta -= delta
        else:
            eta = 0.0
        random_flip = np.random.random_sample(
            (FLAGS.batch_size,FLAGS.seq_length-FLAGS.input_length-1))
        true_token = (random_flip < eta)
        #true_token = (random_flip < pow(base,itr))
        ones = np.ones((FLAGS.img_width//FLAGS.patch_size,
                        FLAGS.img_width//FLAGS.patch_size,
                        int(FLAGS.patch_size**2*FLAGS.img_channel)))
        zeros = np.zeros((FLAGS.img_width//FLAGS.patch_size,
                          FLAGS.img_width//FLAGS.patch_size,
                          int(FLAGS.patch_size**2*FLAGS.img_channel)))
        print("this is ok")
        mask_true = []
        for i in range(FLAGS.batch_size):
            for j in range(FLAGS.seq_length-FLAGS.input_length-1):
                if true_token[i,j]:
                    mask_true.append(ones)
                else:
                    mask_true.append(zeros)
        mask_true = np.array(mask_true)
        mask_true = np.reshape(mask_true, (FLAGS.batch_size,
                                           FLAGS.seq_length-FLAGS.input_length-1,
                                           FLAGS.img_width//FLAGS.patch_size,
                                           FLAGS.img_width//FLAGS.patch_size,
                                           int(FLAGS.patch_size**2*FLAGS.img_channel)))
        print("mask true shape")
        print(mask_true.shape)
        print("begin to train")
        cost = model.train(ims, lr, mask_true)
        if FLAGS.reverse_input:
            ims_rev = ims[:,::-1]
            cost += model.train(ims_rev, lr, mask_true)
            cost = cost/2
            y.append(cost)

        if itr % FLAGS.display_interval == 0:
            print('itr: ' + str(itr))
            print('training loss: ' + str(cost))
            x.append(itr)
        
        
        
        

 #######################       #just for the time being suspend when successfucl debug  can  reply this  module###################################################
       
        if itr % FLAGS.snapshot_interval == 0:
            model.save(itr)

        train_input_handle.next()
    
    plt.figure()
    plt.title("this is loss")
    plt.plot(x,y)
    plt.savefig('/media/workdir/hujh/AlexHu-predrnn/alexhupredrnn/loss.pdf')
    

if __name__ == '__main__':
    tf.app.run()

