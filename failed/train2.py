#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train part 2
use the masked images to train
!!!!!!!!!!!FAILED!!!!!
"""


import numpy as np
from collections import deque

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
#from keras.optimizers import Adam
from numpy import random
import matplotlib.pyplot as plt
import pickle
import os
#import tensorflow as tf

from game_api2 import jump_API

img_rows = 100
img_cols = 100
img_channels = 1
ACTIONS = 15 # how many actions can be taken
MIN_MS = 15 # the minimum steps (press time in old API)
MAX_MS = 69
OBSERVATION = 100 # how many observations before start to train
INITIAL_EPSILON = 0.15
FINAL_EPSILON = 0.0001
EXPLORE = 1000 # how many steps from INITIAL_EPSILON to FINAL_EPSILON
REPLAY_MEMORY = 1000 # number of states to remember
BATCH = 300 # size of minibatch
GAMMA = 0.1 # the decay rate
MONITOR = True # whether to show the images
MASK = True # use the masked image to train # FAILED!!!

def buildmodel(show_model=False):
    model = Sequential()
    model.add(Conv2D(8,(5,5),activation='relu',input_shape=(img_rows,img_cols,img_channels)))
    model.add(MaxPooling2D((4,4)))
    model.add(Conv2D(16,(3,3),activation='relu'))
    model.add(MaxPooling2D((4,4)))
    model.add(Flatten())
    model.add(Dense(32))
    model.add(Dropout(0.4))
    model.add(Dense(ACTIONS))
    model.compile(loss='mse',optimizer='adam')
    if show_model: print(model.summary())
    return model




####### if you run at the first time:
if not os.path.exists('mem.pickle'):
    model = buildmodel(True)
    mem = deque(maxlen = REPLAY_MEMORY) # store the memories
    epsilon = INITIAL_EPSILON
    t=0
else:####### if you continue to run:
    with open('mem.pickle','rb') as f:
        (t,epsilon,mem)=pickle.load(f)
    from keras.models import load_model
    model = load_model('model.h5')
##################################

g = jump_API(MIN_MS,MAX_MS,ACTIONS,MASK) #initialize an API to the game
s_t = g.first_step()

while True: # start to loop
    print('*********************************')
    print('t=%i,epsilon=%f'%(t,epsilon),end='  ')
    if random.random()<=epsilon:
        print('RANDOM MOVE!')
        a_t = random.choice(ACTIONS)
    else:
        print('Move by model.')
        qs = model.predict(s_t)
        a_t = np.argmax(qs)
    # forward one step
    print('Moving...',end=' ')
    s_t1, r_t, die = g.next_step(a_t)
    print('Done.')
    
    # save it to memory
    print('=========')
    print('NEW Memory: \na_t=%i,r_t=%i,die=%i'%(a_t,r_t,die))
    if die:
        mem.append((s_t,a_t, r_t,None,die))
        if MONITOR: plt.imshow(s_t[0,:,:,0],'gray')
        if MONITOR: plt.show()
        print('Then DIE.')
    else:
        mem.append((s_t,a_t, r_t,s_t1,die))
        if MONITOR: plt.imshow(np.concatenate((s_t,s_t1),axis=2)[0,:,:,0],'gray')
        if MONITOR: plt.show()
    print('=========')
    # update epsilon
    if epsilon > FINAL_EPSILON and t > OBSERVATION:
        epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
    if epsilon < FINAL_EPSILON: epsilon = FINAL_EPSILON
    # trian the model if observations are enough
    if t > OBSERVATION:
        # sample a minibatch
        minibatch = random.choice(len(mem), min(BATCH,len(mem)))
        # initialize input and target
        inputs = np.zeros((BATCH, img_rows, img_cols, img_channels))
        targets = np.zeros((BATCH, ACTIONS))
        # fill them
        for i,j in enumerate(minibatch):
            (s0_t,a0_t, r0_t,s0_t1,die0) = mem[j]
            inputs[i:i+1] = s0_t
            targets[i] = model.predict(s0_t)
            if die0:
                targets[i,a0_t] = r0_t # if die, no future rewards
            else:
                Qt1 = model.predict(s0_t1)
                maxQ = np.max(Qt1)
                targets[i,a0_t] = r0_t + GAMMA * maxQ
        # train the model
        print('Training the model...',end=' ')
        loss = model.train_on_batch(inputs,targets)
        print('Done. loss=%f'%loss)
    # iteration
    s_t = s_t1
    t += 1
    # save the model every 10 times
    if t%50 ==0:
        print('saving model...',end=' ')
        model.save('model.h5')
        with open('mem.pickle','wb') as f:
            pickle.dump((t,epsilon,mem),f)
        print('Done.')
    
