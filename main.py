#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
print(tf.__version__)
tf.config.list_physical_devices('GPU')

import numpy as np
import os
import argparse
from util import *
from matplotlib import pyplot as plt
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--batch_size', type=int, default=4, metavar='N', help='Input batch size for training')
parser.add_argument('--numframe', type=int, default=0, metavar='N', help='Input number of frame of training')
parser.add_argument('--each', type=int, default=2, metavar='N', help='How many each to take image sequence')
args = parser.parse_args()

dname = 'immatrix_2D_40x30_f9_e2'
immatrix = np.load(dname+'.npy')
print('load immatrix with size:', immatrix.shape, flush=True)
n = immatrix.shape[-2] 
m = immatrix.shape[-1] 

ljcoef= [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30]

ori_x = [1,0]
ori = [[1, 0], [0, 1], [1, 5], [1, 4], [1, 3], [1, 2], [1, 1], [2, 1], [3, 1], [4, 1]]

def randomize(a, b):
    # generate the permutation index array.
    permutation = np.random.permutation(a.shape[0])
    # shuffle the arrays
    shuffled_a = a[permutation]
    shuffled_b = b[permutation]
    return shuffled_a, shuffled_b

def split_sequences_to3D(sequences, numframe=1, shuffle= True):
    X, y = list(), list()
    for n_key in range(sequences.shape[0]):
        for n_lj in range(sequences.shape[1]):
            lj=ljcoef[n_lj]
            for o in range(sequences.shape[2]):
                s=sin_angle(ori_x,ori[o])
                # gather input and output parts of the pattern
                seq,_ = acc_crack(np.copy(sequences[n_key, n_lj, o])) 
                seq_y = np.copy(seq) 
                seq_y -= 1
                seq_y *=-1
                for f in range(numframe):
                    seq_x = seq[f]
                    seq_x = seq_x[:,:,np.newaxis]
                    seq_x = np.dstack((seq_x, seq_x))
                    seq_x[:,:,0] *= (lj+1)
                    seq_x[:,:,1] *= (s+1)
                    seq_x -= 1
                    X.append(np.copy(seq_x))
                    y.append(np.copy(seq_y))
    X = np.array(X)
    y = np.array(y)
    if(shuffle):
        X, y = randomize(X, y)
    return X, y

if args.numframe == 0:
    numframe = immatrix.shape[3]
else:
    numframe = args.numframe
try:
    X = np.load(dname+'_'+str(numframe)+'_X.npy')
    y = np.load(dname+'_'+str(numframe)+'_y.npy')
except:
    dataset = immatrix#[:-1] #np.load('projected.npy')# hstack((in_seq1, in_seq2, out_seq))
    print('dataset shape:', dataset.shape, flush = True)
    # convert into input/output
    X, y = split_sequences_to3D(dataset, numframe, shuffle = True)
    # the dataset knows the number of features, e.g. 2
    np.save(dname+'_'+str(numframe)+'_X.npy',X)
    np.save(dname+'_'+str(numframe)+'_y.npy',y)
else:
    print('input shape:', X.shape, flush=True)
    print('output shape:', y.shape, flush=True)    


import os
from numpy import array, hstack
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, BatchNormalization, Activation
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import optimizers, losses, metrics
from tensorflow.keras.layers import Dense, Flatten, Reshape, Conv2D, LSTM, Conv3D, Conv3DTranspose 

os.environ['KERAS_BACKEND']='tensorflow'
opt = optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay = 0.001)

stamp=''
activation = 'sigmoid'
loss = tf.keras.losses.BinaryCrossentropy()

metrics=['acc']

batch_size = args.batch_size 
fs = 1024
ks = 3
epochs = 10000
pat = 100


def Conv_LSTM_2to3():
    # define model
    model = Sequential()
    # ----------------------encoder--------------------
    model.add(Conv2D(fs//4, 5, padding="same", input_shape=(n, m, 2), strides=(2,2)))
    model.add(Activation('gelu'))
    model.add(Dropout(0.3))
    
    model.add(Conv2D(fs//4, 3, padding='same', strides=(2,2)))
    model.add(Activation('gelu'))
    model.add(Dropout(0.3))
    
    # ----------------------Attention+LSTM--------------------
    model.add(Reshape((-1, fs//4)))
    model.add(LSTM(fs, return_sequences=True))
    model.add(Dense(75*fs//16))
   
    model.add(Reshape((1, 5, 5, -1)))
    
    # ----------------------decoder--------------------
    model.add(Conv3DTranspose(fs//4, 3, padding='same', strides= (3,2,2)))
    model.add(Activation('gelu'))

    model.add(Conv3DTranspose(fs//4, 5, padding='same', strides= (3,3,4)))
    model.add(Activation('gelu'))

    model.add(Conv3D(1, 3, activation = activation, padding='same'))

    model.compile(optimizer=opt, loss=loss, metrics=metrics)
    model.summary()
    return model



model = Conv_LSTM_2to3()
stamp='40x30_allframes_test'
    
from time import strftime
stamp = stamp+'_'+strftime("%m_%d_%H_%M")

print('stamp:', stamp)

os.makedirs('model/'+stamp, exist_ok=True)

# np.random.seed(9527)
validation_split=0.5

if validation_split !=0:
    target = 'val_'
else:
    target = ''

chkp = ModelCheckpoint('model/'+stamp+'/model.h5', monitor=target+'loss', verbose=1, save_best_only=True, save_weights_only=True)
es_l = EarlyStopping(monitor=target+'loss', patience=pat, verbose=0, restore_best_weights=True)

print('\n', flush=True)

# fit model
history = model.fit(X[:X.shape[0]//2], y[:y.shape[0]//2], validation_split=validation_split
        , epochs = epochs, verbose=2, callbacks=[chkp, es_l]
        , batch_size = batch_size
         )
print('training has finished', flush=True)

model.load_weights('model/'+stamp+'/model.h5')
print('restoring the best checkpoint, and save the model to model/'+stamp+'/model')
model.save('model/'+stamp+'/model', save_format='tf')

plt.figure()
plt.plot(history.history['loss'])
if validation_split!=0:
    plt.plot(history.history['val_loss'])
   
    plt.legend(['train', 'val'])

plt.xlabel('Epoch')
plt.title('model loss')
plt.savefig('model/'+stamp+'/loss.png')

if loss != 'mse':
    plt.figure()
    plt.plot(history.history['acc'])

    if validation_split!=0:
        plt.plot(history.history['val_acc'])
        plt.legend(['train', 'val'])

    plt.xlabel('Epoch')
    plt.title('model accuracy')
    plt.savefig('model/'+stamp+'/acc.png')

