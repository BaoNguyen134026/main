#!/usr/bin/env python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn import svm
from sklearn import metrics
import time
import numpy as np
import pickle
import cv2
import math as m
import matplotlib.pyplot as plt

def motion_kinds(point_detect):
        point_detect = np.array(point_detect)
        detect = np.arange(len(point_detect)).reshape((len(point_detect),1)).tolist()
        for i in range(1,len(point_detect)):
            P15_distance[i] = m.sqrt(m.pow(point_detect[i][0]-point_detect[i-1][0],2)
                                    +m.pow(point_detect[i][1]-point_detect[i-1][1],2)
                                    +m.pow(point_detect[i][2]-point_detect[i-1][2],2))
        b = [i for i in P15_distance if i >0.01]
        # print('len(b) =',len(b))
        if len(b) > 11:
            for i in range(0,15):
                detect[int(i)] =  [point_detect[int(i)][0] - point_detect[0][0],
                                    point_detect[int(i)][1] - point_detect[0][1],
                                    point_detect[int(i)][2] - point_detect[0][2]]
            detect=np.array(detect)
            detect = np.reshape(detect,(1,45))
            a = loaded_model.predict(detect)
        else: a = None
        return a 
def motion_detection(point_3d):
    global first_loop, cnt, Points_15, Points_3
    if first_loop == True:
        Points_15[cnt - 1] = [point_3d[0],
                            point_3d[1],
                            point_3d[2]]
        cnt+=1
        if cnt >= 15:
            first_loop = False
            cnt = 0
            return motion_kinds(Points_15)
    else:
        if cnt <= 2:
            Points_3[cnt] = [point_3d[0],
                                point_3d[1],
                                point_3d[2]]
            cnt+=1
             
        else:
            for ii in range(0,12):
                Points_15[int(ii)] = [Points_15[int(ii)+3][0],
                                                    Points_15[int(ii)+3][1],
                                                    Points_15[int(ii)+3][2]]
            for ii in range(12,15):
                Points_15[int(ii)] = [Points_3[int(ii)-12][0],
                                    Points_3[int(ii)-12][1],
                                    Points_3[int(ii)-12][2]]
            cnt = 0
            return motion_kinds(Points_15)
        

if __name__ == "__main__":
    try:
        cap1 = cv2.VideoCapture('/home/bao/Downloads/outpy.avi')
        f1 = np.load("/home/bao/Downloads/save3d.npy")
        loaded_model = pickle.load(open('/home/bao/Desktop/video/sp/traindongtac.sav', 'rb'))
        #initialized variable
        first_loop = True
        Points_3 = np.arange(3).reshape((3,1)).tolist()
        Points_15 = np.arange(15).reshape((15,1)).tolist()
        P15_distance = np.arange(15).reshape((15,)).tolist()
        cnt = 0
        #run video
        for index in range(int(cap1.get(7))):
            ret,frame = cap1.read()
            point_3d = f1[int(index)]
            if point_3d is not None:
                k = motion_detection(point_3d)
                if k is not None:
                    print('k=\n',k)
                # pass
            #show video
            cv2.imshow('video',frame)
            time.sleep(0.05)
            if cv2.waitKey(30) ==27 :
                break
    finally:
        cv2.destroyAllWindows()