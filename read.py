import os
import csv
import cv2
import random
import time
import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

DIR = 'data/'
filenames = os.listdir('data')
def MakeLabel(flag, sz):
    res = []
    with open(DIR + flag + '.csv', 'r') as f:
        reader = csv.reader(f)
        flag = 0
        cnt = 0
        for line in reader:
            if not flag:
                flag = 1
                continue
            #if cnt >= 5:
            #    break
            cnt += 1
            #print(line)
            tmp = [line[0], line[1]]
            tmp[1] = tmp[1].replace('A', '0')
            tmp[1] = tmp[1].replace('B', '1')
            tmp[1] = tmp[1].replace('C', '2')
            tmp[1] = int(tmp[1])
            res.append(tmp)
    x, y = [], []
    for i in range(len(res)):
        img = mpimg.imread(DIR + res[i][0])
        new = cv2.resize(img, (sz, sz), interpolation = cv2.INTER_LINEAR)
        x.append(new)
        y.append(res[i][1])
    return x, y
