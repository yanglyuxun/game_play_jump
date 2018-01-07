#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
android screen recorder
"""

import subprocess
from moviepy.editor import VideoFileClip

def shell(cmd):
    result = subprocess.run(cmd, stdout=subprocess.PIPE,shell=True)
    print(result.stdout.decode())

### get the recording
shell('adb devices')
shell('adb shell screenrecord /sdcard/tmp.mp4')
shell('adb pull /sdcard/tmp.mp4')
shell('adb shell rm /sdcard/tmp.mp4')

### edit
v = VideoFileClip('tmp.mp4')
v.save_frame("frame.jpeg", t='0:0:6.0')
v.save_frame("frame.jpeg", t='0:0:20.3')
shell('rm frame.jpeg')

shell('ffmpeg -ss 6 -t 14.3 -i tmp.mp4  screen.mp4')

v = VideoFileClip('screen.mp4')
v1 = v.resize(width=130)
v1.write_gif("screen.gif",fps=5)


#my_clip.write_videofile("movie.mp4",fps=15)