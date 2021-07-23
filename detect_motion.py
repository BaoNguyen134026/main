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

import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd

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

def post_process_depth_frame(depth_frame):
    """
    Filter the depth frame acquired using the Intel RealSense device

    Parameters:
    -----------
    depth_frame          : rs.frame()
                           The depth frame to be post-processed
    decimation_magnitude : double
                           The magnitude of the decimation filter
    spatial_magnitude    : double
                           The magnitude of the spatial filter
    spatial_smooth_alpha : double
                           The alpha value for spatial filter based smoothening
    spatial_smooth_delta : double
                           The delta value for spatial filter based smoothening
    temporal_smooth_alpha: double
                           The alpha value for temporal filter based smoothening
    temporal_smooth_delta: double
                           The delta value for temporal filter based smoothening

    Return:
    ----------
    filtered_frame : rs.frame()
                     The post-processed depth frame
    """
    
    # Post processing possible only on the depth_frame
    assert (depth_frame.is_depth_frame())

    # Available filters and control options for the filters
    decimation_filter = rs.decimation_filter()
    spatial_filter = rs.spatial_filter()
    temporal_filter = rs.temporal_filter()

    filter_magnitude = rs.option.filter_magnitude
    filter_smooth_alpha = rs.option.filter_smooth_alpha
    filter_smooth_delta = rs.option.filter_smooth_delta

    # Apply the control parameters for the filter
    decimation_magnitude=1.0
    spatial_magnitude=2.0
    spatial_smooth_alpha=0.5
    spatial_smooth_delta=20
    temporal_smooth_alpha=0.4
    temporal_smooth_delta=20
    decimation_filter.set_option(filter_magnitude, decimation_magnitude)
    spatial_filter.set_option(filter_magnitude, spatial_magnitude)
    spatial_filter.set_option(filter_smooth_alpha, spatial_smooth_alpha)
    spatial_filter.set_option(filter_smooth_delta, spatial_smooth_delta)
    temporal_filter.set_option(filter_smooth_alpha, temporal_smooth_alpha)
    temporal_filter.set_option(filter_smooth_delta, temporal_smooth_delta)

    # Apply the filters
    filtered_frame = decimation_filter.process(depth_frame)
    filtered_frame = spatial_filter.process(filtered_frame)
    filtered_frame = temporal_filter.process(filtered_frame)
    return filtered_frame

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
    
if __name__ == "__main__":
    try:
        # Configure depth and color streams of the intel realsense
        #...from Camera 1
        config_1 = rs.config()
        config_1.enable_device('046122251324')
        config_1.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
        config_1.enable_stream(rs.stream.color, 848, 480, rs.format.rgb8, 30)
        #...from Camera 2
        config_2 = rs.config()
        config_2.enable_device('108222250284')
        config_2.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
        config_2.enable_stream(rs.stream.color, 848, 480, rs.format.rgb8, 30)

        # Start the realsense pipeline
        #...from Camera 1
        pipeline_1 = rs.pipeline()
        pipeline_1.start(config_1)
        #...from Camera 2
        pipeline_2 = rs.pipeline()
        pipeline_2.start(config_2)

        # Create align object to align depth frames to color frames
        #...from Camera 1
        align_1 = rs.align(rs.stream.color)
        #...from Camera 2
        align_2 = rs.align(rs.stream.color)

        # Get the intrinsics information for calculation of 3D point
        #...from Camera 1
        unaligned_frames_1 = pipeline_1.wait_for_frames()
        frames_1 = align_1.process(unaligned_frames_1)
        depth_frame_1 = frames_1.get_depth_frame()
        depth_intrinsic_1 = depth_frame_1.profile.as_video_stream_profile().intrinsics
        color_1 = frames_1.get_color_frame()
        color_image_1= np.asanyarray(color_1.get_data())
        #...from Camera 2
        unaligned_frames_2 = pipeline_2.wait_for_frames()
        frames_2 = align_2.process(unaligned_frames_2)
        depth_frame_2 = frames_2.get_depth_frame()
        depth_intrinsic_2 = depth_frame_2.profile.as_video_stream_profile().intrinsics
        color_2 = frames_2.get_color_frame()
        color_image_2 = np.asanyarray(color_2.get_data())
        
        # Initialize the cubemos api with a valid license key in default_license_dir()
        skeletrack = skeletontracker(cloud_tracking_api_key="")
        joint_confidence = 0.2

        # Create window for initialisation
        window_name = "cubemos skeleton tracking with realsense D400 series"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL + cv2.WINDOW_KEEPRATIO)

        # Initialize
        out = cv2.VideoWriter('outpy2.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (color_image_1.shape[1],color_image_1.shape[0]))
        # loaded_model = pickle.load(open('byt.sav', 'rb'))
        P15_distance = np.arange(15).reshape((15,)).tolist()
        Points_15 = np.arange(15).reshape((15,1)).tolist()
        Points_3 = np.arange(3).reshape((3,1)).tolist()

        pc1 =  rs.pointcloud()
        pc2 =  rs.pointcloud()

        cnt = 0
        first_loop = True
        sle = 0
        while True:
            # Create a pipeline_1 object. This object configures the streaming camera and owns it's handle
            #...from Camera 1
            unaligned_frames_1 = pipeline_1.wait_for_frames()
            frames_1 = align_1.process(unaligned_frames_1)
            depth_frame_1 = frames_1.get_depth_frame()
            color_1 = frames_1.get_color_frame()
            #...from Camera 2
            unaligned_frames_2 = pipeline_2.wait_for_frames()
            frames_2 = align_2.process(unaligned_frames_2)
            depth_frame_2 = frames_2.get_depth_frame()
            color_2 = frames_2.get_color_frame()

            if not depth_frame_1 or not depth_frame_2 or not color_1 or not color_2:
                continue
            # Convert images to numpy arrays
            #...from camera 1
            depth_image_1 = np.asanyarray(depth_frame_1.get_data())
            color_image_1 = np.asanyarray(color_1.get_data())
            color_image_1 = cv2.cvtColor(color_image_1, cv2.COLOR_BGR2RGB)
            #...from camera 2
            depth_image_2 = np.asanyarray(depth_frame_2.get_data())
            color_image_2 = np.asanyarray(color_2.get_data())
            color_image_2 = cv2.cvtColor(color_image_2, cv2.COLOR_BGR2RGB)
            # doc database
            # sle = ?
            
            if sle == 1:
                '''find size'''
                # perform inference and update the tracking id
                skeletons = skeletrack.track_skeletons(color_image_1)

                # render the skeletons on top of the acquired image and display it
                cm.render_result(skeletons, color_image_1, joint_confidence)
                P3d_Skeletons = render_ids_3d(  color_image_1,
                                                skeletons,
                                                depth_frame_1,
                                                depth_intrinsic_1,
                                                joint_confidence)
                #calculate pcd
                points_1 = pc1.calculate(depth_frame_1)
                points_2 = pc2.calculate(depth_frame_2)
                v1 = points_1.get_vertices()
                v2 = points_2.get_vertices()
                verts1 = np.asanyarray(v1).view(np.float32).reshape(-1, 3)  # xyz
                verts2 = np.asanyarray(v2).view(np.float32).reshape(-1, 3)  # xyz

                pcl1 = o3d.geometry.PointCloud()
                pcl1.points_1 = o3d.utility.Vector3dVector(verts1)
                pcl1 = pcl1.voxel_down_sample(voxel_size=0.017)

                pcl2 = o3d.geometry.PointCloud()
                pcl2.points_2 = o3d.utility.Vector3dVector(verts2)
                pcl2 = pcl1.voxel_down_sample(voxel_size=0.017)
                # calculate size
                h,vong = icp(pcl1, pcl2, P3d_Skeletons)
                # ghi sle = 2
            elif sle == 2:
                '''tuong tac'''
                # ghi lien tuc doc tac vao database
                pass
            cv2.imshow(window_name, color_image_1)
            cv2.imshow('2',color_image_2)
            if cv2.waitKey(1) == 27:
                break
        pipeline_1.stop()
        pipeline_2.stop()
        cv2.destroyAllWindows()
        #sle = 0
    except Exception as ex:
        print('Exception occured: "{}"'.format(ex))

