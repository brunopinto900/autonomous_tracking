#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 17:49:02 2023

@author: bruno
"""
# ffmpeg -framerate 10 -pattern_type glob -i "*.png" -c:v libx264 -r 10 output.mp4
#!/usr/bin/env python3

import shutil
import os
import subprocess

sourcedir = os.getcwd()+"/images/" 
prefix = "image"
extension = "png"

files = [(f, f[f.rfind("."):], f[:f.rfind(".")].replace(prefix, "")) for f in os.listdir(sourcedir) if f.endswith(extension)]
maxlen = len(max([f[2] for f in files], key = len))

for item in files:
    zeros = maxlen - len(item[2])
    shutil.move(sourcedir+"/"+item[0], sourcedir+"/"+prefix+str(zeros*"0"+item[2])+item[1])
