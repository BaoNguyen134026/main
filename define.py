#!/usr/bin/env python3
from collections import namedtuple
import util as cm
import cv2
import time
import pyrealsense2 as rs
import math
import numpy as np
from skeletontracker import skeletontracker

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn import svm
from sklearn import metrics
import pickle
import cv2
import open3d as o3d
import matplotlib.pyplot as plt



def render_ids_3d(
    render_image, skeletons_2d, depth_map, depth_intrinsic, joint_confidence):
    thickness = 1
    text_color = (255, 255, 255)
    rows, cols, channel = render_image.shape[:3]

    distance_kernel_size = 5
    
    for skeleton_index in range(len(skeletons_2d)):
        if skeletons_2d[skeleton_index] == []:
            break
        
        skeleton_2D = skeletons_2d[skeleton_index]
        joints_2D = skeleton_2D.joints
        did_once = False
        i=0
        cnt = 0
        save = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        for joint_index in range(len(joints_2D)):
            if did_once == False:
                cv2.putText(
                    render_image,
                    "id: " + str(skeleton_2D.id),
                    (int(joints_2D[joint_index].x), int(joints_2D[joint_index].y - 30)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    text_color,
                    thickness,
                )
                did_once = True
            # check if the joint was detected and has valid coordinate
            if skeleton_2D.confidences[joint_index] > joint_confidence:
                
                distance_in_kernel = []
                low_bound_x = max(
                    0,
                    int(
                        joints_2D[joint_index].x - math.floor(distance_kernel_size / 2)
                    )
                )

                upper_bound_x = min(
                    cols - 1,
                    int(joints_2D[joint_index].x + math.ceil(distance_kernel_size / 2)),
                )

                low_bound_y = max(
                    0,
                    int(
                        joints_2D[joint_index].y - math.floor(distance_kernel_size / 2)
                    ),
                )
                upper_bound_y = min(
                    rows - 1,
                    int(joints_2D[joint_index].y + math.ceil(distance_kernel_size / 2)),
                )
                for x in range(low_bound_x, upper_bound_x):
                    for y in range(low_bound_y, upper_bound_y):
                        distance_in_kernel.append(depth_map.get_distance(x, y))
                median_distance = np.percentile(np.array(distance_in_kernel), 50)
                depth_pixel = [
                    int(joints_2D[joint_index].x),
                    int(joints_2D[joint_index].y),
                ]
                if median_distance >= 0.3:
                    point_3d = rs.rs2_deproject_pixel_to_point(
                        depth_intrinsic, depth_pixel, median_distance
                    )
          
                    point_3d = np.round([float(i) for i in point_3d], 3)
                    

                    point_str = [str(x) for x in point_3d]
                    i = "{}".format(joint_index)
                    
                    cnt +=1
                    save[cnt-1] = [point_3d[0],
                                        point_3d[1],
                                        point_3d[2]]
                    
                    cv2.putText(
                        render_image,
                    
                        str(i) + ' ' + str(point_3d) ,
                        (int(joints_2D[joint_index].x), int(joints_2D[joint_index].y)),
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.4,
                        text_color,
                        thickness,
                    )
            
        if cnt == 18:
            return save
        else: return None