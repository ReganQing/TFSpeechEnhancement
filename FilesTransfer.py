#!usr/bin/python3
# -*- coding: UTF-8 -*-
# Author: Ron
# Date: 2023.2.13 14:44
# Software: Pycharm

import os
import shutil

# get files from current filepath and sort them
path = r'/home/ron/data_aishell/wav/train'
dirs = os.listdir(path)
dirs = sorted(dirs)

# create a folder to store ".wav" files
new_path = r"/home/ron/SpeechEnhancement/Train/CleanVoice/"
if not os.path.exists(new_path):
    os.makedirs(new_path)

# traverse folders
for dir in dirs:
    # get file name in folder
    dir_path = os.path.join(path, dir)
    # print(dir_path)
    files = os.listdir(path=dir_path)
    # print(files)
    for file in files:
        # judge if file name ends with ".wav"
        if file.endswith(".wav"):
            # get absolute path of file
            file_path = os.path.join(dir_path, file)
            # move file to new folder
            shutil.move(file_path, new_path)
        # pass if folder is empty
        else:
            continue


print("文件转移完成")
