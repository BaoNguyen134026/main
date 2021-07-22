import math as m
import time
import cv2
import numpy as np
import pyrealsense2.pyrealsense2 as rs
import open3d as o3d
import copy
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd

def calculate_size(pcd,skeleton):
    # crop xy tu pcd
    '''co the crop bang json hoac xyz:
    - neu la json thi co the tao 1 bien json
    chu khong can phai tao 1 file rieng, nhu sau:
        bounding_polygon = np.array([
        #Vertics Polygon 1
                [488.8989868164062, 612.208984375, 286.5320129394531],
                [485.114990234375, 612.208984375, 286.5320129394531],
                [485.114990234375, 605.0880126953125, 286.5320129394531],
                [488.8989868164062, 605.0880126953125, 286.5320129394531],
        #Vertics Polygon2
                [488.89898681640625, 612.208984375, 291.6619873046875], 
                [485.114990234375, 612.208984375, 291.6619873046875], 
                [485.114990234375, 605.0880126953125, 291.6619873046875],
                [488.89898681640625, 605.0880126953125, 291.6619873046875]]).astype("float64") 
        vol = o3d.visualization.SelectionPolygonVolume()
        vol.orthogonal_axis = "Y"
        vol.axis_max = 1000
        vol.axis_min = -1000
        vol.bounding_polygon = o3d.utility.Vector3dVector(bounding_polygon)
    - neu la xyz thi hoi lag vs dai'''
    # tao key dat diem de sorted
    def func(xy):
        return xy[0]
    #lay 20 diem trong xy1  va sap xep theo thu tu tu lon den nho
    xy1 = sorted(xy1[:20], key = func)
    xy2 = sorted(xy2[:20], key = func)
    xy3 = sorted(xy3[:20], key = func)
    # lay x y  ra de tim ham`
    x1,y1,x2,y2,x3,y3 = [],[],[],[],[],[]
    for i in range(19):
        x1.append([xy1[i][0]])
        y1.append([xy1[i][1]])
        x2.append([xy2[i][0]])
        y2.append([xy2[i][1]])
        x3.append([xy3[i][0]])
        y3.append([xy3[i][1]])
    #############################Caculating ##################################
    poly = PolynomialFeatures(degree = 8)

    X1_poly = poly.fit_transform(x1) # x->poly
    X2_poly = poly.fit_transform(x2)
    X3_poly = poly.fit_transform(x3)
    poly.fit(X1_poly, y1) # y->poly co x
    poly.fit(X2_poly, y2)
    poly.fit(X3_poly, y3)

    lin1 = LinearRegression()
    lin2 = LinearRegression()
    lin3 = LinearRegression()
    lin1.fit(X1_poly, y1)
    lin2.fit(X2_poly, y2)
    lin3.fit(X3_poly, y3)

    y_1 = lin1.predict(poly.fit_transform(x1)) # y theo x
    y_2 = lin2.predict(poly.fit_transform(x2))
    y_3 = lin3.predict(poly.fit_transform(x3))
    lenght = [0,0,0]
    for i in range(18):
        lenght[0] = lenght[0]+ m.sqrt((x1[i+1][0]-x1[i][0])**2+((y_1[i+1][0])-y_1[i][0])**2)
        lenght[1] = lenght[0]+ m.sqrt((x2[i+1][0]-x2[i][0])**2+((y_2[i+1][0])-y_2[i][0])**2)
        lenght[2] = lenght[0]+ m.sqrt((x3[i+1][0]-x3[i][0])**2+((y_3[i+1][0])-y_3[i][0])**2)
    #khuc nay tinh chieu cao dua vo cai nen, de t hoi thuy lai sau
    pcd = o3d.io.read_point_cloud("plan0.pcd")
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                            ransac_n=3,
                                            num_iterations=1000)
    [a, b, c, d] = plane_model
    hight = abs((a*tophead[0]+b*tophead[1]+c*tophead[2]+d)/m.sqrt(a**2+b**2+c**2))
    # tra ve h,l
    return hight,lenght
pcd1 = o3d.io.read_point_cloud('test0.ply')
pcd2 = o3d.io.read_point_cloud('test1.ply')
# se dua 2 cai pcd vaf skeleton vao ham icp de xu ly
def icp(pcd1,pcd2,skeleton):
    # dau tien la down_sample pcd
    pcd1 = pcd1.voxel_down_sample(voxel_size=0.001)
    pcd2 = pcd2.voxel_down_sample(voxel_size=0.001)
    
    #crop background here
    '''crop truc x
    z = -2.136 +- 0.3
    y = +- 0.3
    '''
    # lay file json ra
    vol = o3d.visualization.read_selection_polygon_volume("cropped.json")
    # cat background cua 2 pcd bang json
    pcd1 = vol.crop_point_cloud(pcd1)
    pcd2 = vol.crop_point_cloud(pcd2)
    # find plane and delete
    # tim 2 mat phang cua pcd1 va pcd2
    plane_model, inliers = pcd1.segment_plane(distance_threshold=0.03,
                                            ransac_n=3,
                                            num_iterations=1000)
    plane_model, inliers = pcd2.segment_plane(distance_threshold=0.03,
                                                ransac_n=3,
                                                num_iterations=1000)
    # khuc nay la lay pcd mat phang ra k can thiet lam
    inlier_cloud = pcd1.select_by_index(inliers)
    inlier_cloud = pcd2.select_by_index(inliers)
    # khuc nay lay nguoi da loai mat phang
    pcd1 = pcd1.select_by_index(inliers, invert=True)
    pcd2 = pcd2.select_by_index(inliers, invert=True)
    #icp
    threshold = 0.02
    # ma tran doi he truc
    trans_init = np.asarray([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0.0, 0.0, 0.0, 1.0]])
    # tim cai ma tran doi dung
    reg_p2l = o3d.pipelines.registration.registration_icp(
        pcd2, pcd1, threshold, trans_init, 
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    # doi cai pcd2 qua pcd1
    pcd2.transform(reg_p2l.transformation)
    #r gop 2 cai pcd lai
    newpointcloud = pcd2 + pcd1
    #giam so luong diem o cho trung cua 2 pcd
    pcd = newpointcloud.voxel_down_sample(voxel_size=0.02)

    # find noise
    # khuc nay phuc tap lam, noi khu noise la duoc roi 
    # http://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html#Hidden-point-removal
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(
            pcd.cluster_dbscan(eps=0.05, min_points=20, print_progress=False))
    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")

    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0  # colors: <class 'numpy.ndarray'>
    # tra ve ham 
    return calculate_size(pcd,skeleton)
    

