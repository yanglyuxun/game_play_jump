#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Android API by uiautomator with abd tools
imply color select

notes:
    
jump range: 300-1200
"""

import subprocess
#from keras.preprocessing import image
from PIL import Image
import numpy as np
import cv2
#import matplotlib.pyplot as plt
import pickle
import time
from uiautomator import device as d

def shell(cmd):
    result = subprocess.run(cmd, stdout=subprocess.PIPE,shell=True)
    return result.stdout.decode()

def screen_shot():
    'screenshot and return colorful and gray images'
    img_bytes = d.screenshot(None)
    img = Image.open(Image.io.BytesIO(img_bytes))
    return img, img.convert('L')

def image2array(img):
    'trans image object to a [0,1] range array'
    x = np.array(img)
    x = x.reshape((1,x.shape[0],x.shape[1],-1))      
    return x/255

def get_state(img, resize=(100,250), crop=(0,85,100,185)):
    'get the state area image, and trans to array'
    if resize: img = img.resize(resize)
    if crop: state = img.crop(crop)
    return image2array(state)

def mask_back(img_color): # a failed function!!
    'mask the area of the small character'
    state = get_state(img_color)
    state = state[0]
    #plt.imshow(state[-10:,:10])
    c1=state[:10,:10].mean(axis=0).mean(axis=0)
    c2=state[:10,-10:].mean(axis=0).mean(axis=0)
    c3=state[-10:,:10].mean(axis=0).mean(axis=0)
    c4=state[-10:,-10:].mean(axis=0).mean(axis=0)
    c = np.stack((c1,c2,c3,c4))
    cmax = c.max(0)
    cmin = c.min(0)
    delta = cmax-cmin
    cmax += 1*delta
    cmin -= 1*delta
    mask = cv2.inRange(state, cmin, cmax)
    mask = 1.-mask/255.
    mask = mask.reshape((1,mask.shape[0],mask.shape[1],1))
    return mask
    
    
def get_score(img, crop=(0,0.1,1,0.16)):
    'get the score area image'
    w,h = img.size
    crop = (crop[0]*w,crop[1]*h,crop[2]*w,crop[3]*h)
    if crop: score = img.crop(crop)
    return score


def split_score(img):
    '''
    input the image object
    output the splited numbers arrays (range:0,255)
    '''
    im = np.array(img)
    im = cv2.threshold(im,127,255,cv2.THRESH_BINARY_INV)[1]
    cols = (im.sum(axis=0)>10).astype(int)
    dif=np.diff(cols)
    l = np.where(dif==1)[0]+1
    r = np.where(dif==-1)[0]+1
    ims = []
    for l0,r0 in zip(l,r):
        ims.append(im[:,l0:r0])
    return ims

def get_number_samples(samples={}):
    'used to make the number samples'
    while True:
        n = input('Input the score:')
        img = get_score(screen_shot())
        ims = split_score(img)
        assert len(n)==len(ims)
        for i,n0 in enumerate(n):
            samples[int(n0)] = ims[i]
        print('now we have',list(samples.keys()))
        if len(samples)==10:
            break
#    h = max(s.shape[0] for s in samples.values())
#    w = max(s.shape[1] for s in samples.values())
#    for n in samples:
#        ori = samples[n]
#        samples[n] = np.zeros((h,w),dtype='uint8')
#        samples[n][:ori.shape[0],:ori.shape[1]] = ori
    with open('samples.pickle','wb') as f:
        pickle.dump(samples,f)
    return samples

def detect_score(screen_img, samples):
    'detect the score in the screen'
    try:
        score_img = get_score(screen_img)
        ims = split_score(score_img)
        groupOutput = []
        for im in ims:
            scores = []
            h,w = samples[0].shape
            new_im = np.zeros((h+5,w+5),dtype='uint8')
            new_im[:im.shape[0],:im.shape[1]]=im
            im = new_im[:h,:w]
            for (digit, digitROI) in samples.items():
                result = cv2.matchTemplate(im,digitROI,cv2.TM_CCOEFF)
                (_, score, _, _) = cv2.minMaxLoc(result)
                scores.append(score)
            groupOutput.append(str(list(samples.keys())[np.argmax(scores)]))
        return int(''.join(groupOutput))
    except:
        return None

def has_die(img):
    'input image, output if this is the die screen'
    x = image2array(img)
    if (x[0][0][0][0] < 0.4) and (x[0][0][len(x[0][0]) - 1][0] < 0.4) and (
        x[0][len(x[0]) - 1][0][0] < 0.4) and (x[0][len(x[0]) - 1][len(x[0][0]) - 1][0] < 0.4):
        return True
    else:
        return False

def restart():
    'press the restart button'
    d.swipe(540,1590,540,1590,1)
    #t=shell('adb shell input swipe 540 1590 540 1590 10')
    #if t: print(t)

def jump(dur):
    'press the screen for press_time ms'
    x,y = 500+ np.random.choice(50), 1600+ np.random.choice(50)
    cmd = 'adb shell input swipe {x1} {y1} {x2} {y2} {duration}'.format(
        x1=x ,
        y1=y,
        x2=x,
        y2=y,
        duration = dur)
    t=shell(cmd)
    if t: print(t)

class jump_API(object):
    'the API to control the game'
    def __init__(self, minms=15,maxms=69,actions=15,mask=False):
        '''
        minms = the minimum number of ms to press
        maxms = the maximum number of ms to press
        actions = the number of actions
        '''
        self.minms = minms
        self.maxms = maxms
        self.actions = actions
        self.mask = mask
        d = maxms - minms
        interval = d/(actions-1)
        self.diff = interval
        print('Minimum difference between actions: %i ms'%interval)
        a_map = [minms+interval*i for i in range(actions)]
        self.a_map = [int(i) for i in a_map]
        self.maxscore = 0
        with open('samples.pickle','rb') as f:
            self.samples = pickle.load(f)
        print('ADB devices:')
        print(shell('adb devices'))
    def detect_sc(self, img):
        # to avoid the following to be None
        return detect_score(img,self.samples) or self.score_t
    def first_step(self):
        'get the infomation of first step'
        self.img = screen_shot()
        self.if_die_restart()
        self.s_t = get_state(self.img[1])
        self.score_t = 0
        self.score_t = self.detect_sc(self.img[1])
        if self.mask:
            return mask_back(self.img[0])
        else:
            return self.s_t
    def if_die_restart(self):
        'if current img is die, restart until complished'
        while True:
            if not has_die(self.img[1]):
                time.sleep(0.8) # to let the toy to go to the ground
                self.img = screen_shot()
                break
            else:
                restart()
                #time.sleep(0.2)
                self.img = screen_shot()
    def next_step(self,action, wait = 1):
        '''
        forward a step from the first step
        action is an int in range(actions)
        wait = 2 : wait 2 seconds after each jump
        '''
        assert action < self.actions
        ms = self.a_map[action]
        jump(ms)
        #time.sleep(wait) # wait some seconds
        t00 = time.time()
        img1 = screen_shot()
        # if has not die, but score does not increase, wait 3 seconds
        t0=time.time()
        while time.time()<t0+3 and (not has_die(img1[1])) and self.detect_sc(img1[1])==self.score_t:
            img1 = screen_shot()
        if time.time()-t00<0.5:
            time.sleep(0.5)
        img1 = screen_shot()
        die = has_die(img1[1])
        if die:
            r_t = 0
            self.maxscore = max(self.maxscore, self.score_t)
            print('Dead. Score: %i'%self.score_t)
            print('history max score: %i'%self.maxscore)
            print('Now restarting...')
            self.first_step() # restart and record info
            assert self.score_t==0 # restarted and the score should be 0
        else:
            score_t1 = self.detect_sc(img1[1])
            r_t = score_t1 - self.score_t # reward
            self.score_t = score_t1
            self.img = img1
            self.s_t = get_state(self.img[1])
        if self.mask:
            return mask_back(self.img[0]),r_t, die
        else:
            return self.s_t, r_t, die
            

