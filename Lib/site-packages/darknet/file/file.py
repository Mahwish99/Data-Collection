"python3.5"

import os
import shutil

"""
path  :
suffix:
"""
def get_all_file_name(path, suffix=None):
    if os.path.exists(path) == False:
        print("[Error]:%s no exist!"%(path))
        return []
    filelist = []
    files = os.listdir(path)
    for f in files:
        if os.path.isdir(f) == True:
            continue
        if suffix is None:
            filelist.append(f)
            continue
        strs = f.split(".")
        sf = strs.pop()
        if sf in suffix:
            filelist.append(f)
    return filelist

"""
path  :
"""
def get_all_sub_directory_name(path):
    if os.path.exists(path) == False:
        print("[Error]:%s no exist!"%(path))
        return []
    dirlist = []
    files = os.listdir(path)
    for f in files:
        if os.path.isdir() == False:
            continue
        dirlist.append(f)
    return dirlist

"""
path   :
dirname:
"""
def make_directory(path, dirname):
    if isinstance(dirname, list) == False:
        if os.path.exists(path+"/"+dirname) == True:
            print("[Error]:directory already exist!")
            return 1
        os.mkdir(path+"/"+dirname)
        return 0
    flg = False
    for elem in dirname:
        if os.path.exists(path+"/"+elem) == True:
            flg = True
            print("[Error]:directory (%s) already exist!"%(elem))
            continue
        os.mkdir(path+"/"+elem)
    if flg == True:
        return 1
    return 0

"""
path   :
dirname:
"""
def remove_directory(path, dirname):
    if isinstance(dirname, list) == False:
        if os.path.exists(path+"/"+dirname) == False:
            print("[Error]:directory no exist!")
            return 1
        shutil.rmtree(path+"/"+dirname)
        return 0
    flg = False
    for elem in dirname:
        if os.path.exists(path+"/"+elem) == False:
            flg = True
            print("[Error]:directory (%s) no exist!"%(elem))
            continue
        shutil.rmtree(path+"/"+elem)
    if flg == True:
        return 1
    return 0

"""
path   :
filename:
"""
def remove_files(path, filename):
    if isinstance(filename, list) == False:
        if os.path.exists(path+"/"+filename) == False:
            print("[Error]:file no exist!")
            return 1
        os.remove(path+"/"+filename)
        return 0
    flg = False
    for elem in filename:
        if os.path.exists(path+"/"+elem) == False:
            flg = True
            print("[Error]:file (%s) no exist!"%(elem))
            continue
        os.remove(path+"/"+elem)
    if flg == True:
        return 1
    return 0

