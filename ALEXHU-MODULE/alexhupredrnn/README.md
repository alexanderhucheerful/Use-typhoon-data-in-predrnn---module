# PredRNN++  and  typhondata provider application
This is a TensorFlow implementation of [PredRNN++](https://arxiv.org/abs/1804.06300), a recurrent model for video prediction as described in the following paper:

**PredRNN++: Towards A Resolution of the Deep-in-Time Dilemma in Spatiotemporal Predictive Learning**, by Yunbo Wang, Zhifeng Gao, Mingsheng Long, Jianmin Wang and Philip S. Yu.
 upper description is this module's author comment,if you apply this module in business,you need to Contact the author rather than me.
 and  i  modifiy this module and useing typhon path data to train and predict ,if you want to use typhon data and op this module you must read comment as flow.

## Setup
Required python libraries: tensorflow (>=1.0) + opencv + numpy.\
Tested in ubuntu/centOS + nvidia titan X (Pascal) with cuda (>=8.0) and cudnn (>=5.0).
beside you need matplotlib although this is a nonsense

## Datasets
you need download data from  typhoon_data_list_20190730.txt 
and train data into trainpng,test data into round1_testsetpng

## Training
you need enter folder of alexhupredrnn
Use the alexhu_train.py script to train the model. To train the default model on typhon datasets simply use:
```
python alexhu_train.py
```
and for save my time you need change some paths that is crucial for module ,you should discover by your self

## Prediction samples
```
python alexhupred.py
```



