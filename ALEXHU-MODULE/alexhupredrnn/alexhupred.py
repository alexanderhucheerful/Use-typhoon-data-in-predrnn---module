from __future__ import print_function

import os, time, scipy.io, shutil

import numpy as np

import PIL.Image as Image

import cv2 as cv

import glob

import argparse

import re

import tensorflow as tf

from alexhu_train import *

import matplotlib.pyplot as plt 

##################################################################################################
# this list is test data key ---the wave band number                                             #
# in order to save my time so i brief some process you need change the dir of all location       #
##################################################################################################
HEADS = ['U']
#this dir is your test data perserve path
test_dir = '/media/workdir/hujh/AlexHu-predrnn/round1_testsetpng/'
save_dir = '/media/workdir/hujh/AlexHu-predrnn/result/'

for i  in range (1):
    module = Model()
    #new_saver = tf.train.import_meta_graph('/media/workdir/hujh/AlexHu-predrnn/checkpoint/model.ckpt-330.meta')
    module.saver.restore(module.sess,tf.train.latest_checkpoint('/media/workdir/hujh/AlexHu-predrnn/checkpoint'))
    #graph = tf.get_default_graph()


    for head in HEADS:

        print('Predicting:', head)

        filename = head + '_Hour_*.png'

        files = glob.glob(os.path.join(test_dir, filename))

        tids = []

        for i in range(len(files)):

            tids.append(int(re.findall(r'Hour_(\d+)', files[i])[0]))

        last_tid = max(tids)



        framess= []

        for past_tid in range(last_tid - 5, last_tid + 1):

            frame = np.array(Image.open(os.path.join(test_dir, head + '_Hour_' + str(past_tid) + '.png'))) / 255.0
            frame = cv.resize(frame,(20,20),interpolation=cv.INTER_CUBIC)

            #frame = np.transpose(frame.astype(np.float32), axes=[2, 0, 1])

            framess.append(frame)
        
        framess = np.array(framess)
        frames = []
        frames = np.concatenate((framess,framess),axis =0)
        frames = np.array(frames)
        print('the frames shape is :')
        print(frames.shape)


        mask_true = np.zeros((FLAGS.batch_size,
                              FLAGS.seq_length - FLAGS.input_length -1,
                              FLAGS.img_width,
                              FLAGS.img_width,
                              FLAGS.patch_size* FLAGS.img_channel))




        frames = frames[np.newaxis,:,:,:,:]
        output = module.test(frames,mask_true)
        output = np.array(output)

finaloutput = []
output = np.squeeze(output)
print("output size is :")
print(output.shape)
finaloutput = np.concatenate((framess,output),axis =0)
finaloutput = np.array(finaloutput)

fig = plt.figure()
subplot = (3,4,12)
for i in range(1,13,1):
    
    ax = fig.add_subplot(1,12,i)
    plt.axis('off')
    plt.subplots_adjust(left=0.04, top= 0.90, right = 0.96, bottom = 0.04, wspace = 0, hspace = 0)
    if i ==7:
        plt.title(str(head)+' channel'+' INPUT 6 PRED 6')
    ax.imshow(finaloutput[i-1])


fuckname = '/media/workdir/hujh/AlexHu-predrnn/result/' + str(head)+'.png'
plt.subplots_adjust(left=0.04, top= 0.90, right = 0.96, bottom = 0.04, wspace =0, hspace = 0)
#plt.title('INPUT 6 PRED 6')
plt.savefig(fuckname,dpi=400,pad_inches=0.0)
plt.show()
    
 



  


