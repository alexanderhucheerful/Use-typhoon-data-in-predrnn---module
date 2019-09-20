import numpy as np
import random
import cv2 as cv
import glob 
import os
import re
import PIL.Image as Image

class InputHandle:
    def __init__(self, input_param):
        self.paths = input_param['paths']
        self.name = input_param['name']
        self.input_data_type = input_param.get('input_data_type', 'float32')
        self.output_data_type = input_param.get('output_data_type', 'float32')
        self.minibatch_size = input_param['minibatch_size']
        self.is_output_sequence = input_param['is_output_sequence']
        #on the base of cv.resize
        self.image_width = 20
        self.fnum = 12
        self.freq = 6
        self.image_channel = 3
        self.current_position = 0
        self.current_input_length = 6
        self.current_output_length = 6
        #self.load()

    # for simple process and save my study time so i defualt set the module is not shuffle
    def begin(self):
        self.current_batch_size = self.minibatch_size
        if self.paths == '/media/workdir/hujh/AlexHu-predrnn/round1_testsetpng/':
            self.sid = 'U'
        else:             
            self.sid = 'A'
        filename = self.sid + '_Hour_*.png'
        print(filename)
        print(self.paths)
        self.files = glob.glob(os.path.join(self.paths, filename))
        print(self.files)
        

        self.allframes = [None] * len(self.files)
        tids = []
        for i in range(len(self.allframes)):
            self.allframes[i] = []
            tids.append(int(re.findall(r'Hour_(\d+)', self.files[i])[0]))
    
        self.first_tid = min(tids)
        self.last_tid = max(tids)
        self.tids = sorted(tids)
        
    def initial(self):
    	  self.current_position = 0
      

    def __len__(self):
        lengths = len(self.files) - (self.fnum // 2) * (self.freq + 1) + 1
        print("the lengths is :")
        print(lengths)
        return lengths


    def _get_patch(self,images):
        images = cv.resize(images, (self.image_width, self.image_width), interpolation=cv.INTER_CUBIC)
        return images


    def __getitem__(self, idx):
        tid = self.tids[idx]
        ids1 = list(range(tid, tid + (self.fnum // 2)))
        ids2 = list(range(tid + (self.fnum // 2) + self.freq - 1, tid + (self.fnum // 2) + self.freq + (self.fnum // 2) * self.freq - 1, self.freq))
        ids = ids1 + ids2

        frames = []
        for ctid in ids:
            if not len(self.allframes[ctid - self.first_tid]):
                frame = np.array(Image.open(os.path.join(self.paths, self.sid + '_Hour_' + str(ctid) + '.png'))) / 255.0
                frame = self._get_patch(frame)
                #frame = np.transpose(frame.astype(np.float32), axes=[2, 0, 1])
                self.allframes[ctid - self.first_tid] = frame
            else:
                frame = self.allframes[ctid - self.first_tid]
            frames.append(frame)

            frames_crop = frames
            frames_crop = np.array(frames_crop)
            #print("the frames_crop shape is")
            #print(frames_crop.shape)
            #print(frames_crop)
            #print(ids)
        return ids, frames_crop

    def _get_IdList(self):
        self.id = np.arange(self.lengths)
        return self.id

    def next(self):
        self.current_position += 1

    def no_batch_left(self):
        if self.current_position <self.__len__()-1:
            return True
        else:
            return False
    def get_batch(self):
        #input_seq = self.input_batch()
        #output_seq = self.output_batch()
        #batch = np.concatenate((input_seq, output_seq), axis=1)
        temp = []
        for i in range(self.minibatch_size):
            print("this batch size is :")
            print(self.minibatch_size)
            ids,frames_crop = self.__getitem__(int(self.current_position+i))
            print("the current_position is :")
            print(int(self.current_position+i))
            temp.append(frames_crop[:,:,:,:])
          
        temp = np.array(temp)
        #print("the temp.shape is:")
        #print(temp.shape)
        batch = temp
        batch = batch.astype(self.input_data_type)
        print("the  batch shape is cporret?")
        print(batch.shape)
        
        return batch










       