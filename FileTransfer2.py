#!usr/bin/python3
# -*- coding: UTF-8 -*-
# Author: Ron
# Date: 2023.2.13 16:57
# Software: Pycharm

import os
import shutil

# get files from current filepath and sort them
path = r'/home/ron/data_aishell/wav/train/'

dirs = os.listdir(path)
dirs = sorted(dirs)

# create a folder to store ".wav" files
new_path = r"/home/ron/SpeechEnhancement/Train/CleanVoice/"

if not os.path.exists(new_path):
    os.makedirs(new_path)

# traverse folders
for i in range(5000):
    if dirs[i].endswith(".wav"):
        # get absolute path of file
        dir_path = os.path.join(path, dirs[i])
        # move file to new folder
        shutil.move(dir_path, new_path)
    # pass if folder is empty
    else:
        continue


print("文件转移完成")
