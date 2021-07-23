from flask import Flask, render_template, Response
import cv2
import time
# import mysql.connector
import pyrealsense2 as rs
from collections import namedtuple
import util as cm
import math
import numpy as np
from skeletontracker import skeletontracker

app = Flask(__name__)

# mydb = mysql.connector.connect(
#   host="localhost",
#   user="byt",
#   password="1231",
#   database="interaction"
# )
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
def gen_frames(camera_id):
    if int(camera_id) == 0:
        config_1 = rs.config()
        config_1.enable_device('108222250284')
        config_1.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
        config_1.enable_stream(rs.stream.color, 848, 480, rs.format.rgb8, 30)
        colorizer = rs.colorizer()
        # Start the realsense pipeline
        pipeline_1 = rs.pipeline()
        pipeline_1.start(config_1)
        # Create align object to align depth frames to color frames
        align_1 = rs.align(rs.stream.color)
        for i in range(30):
            pipeline_1.wait_for_frames()
        # Get the intrinsics information for calculation of 3D point
        unaligned_frames_1 = pipeline_1.wait_for_frames()
        frames_1 = align_1.process(unaligned_frames_1)
        depth_1 = frames_1.get_depth_frame()
        depth_intrinsic_1 = depth_1.profile.as_video_stream_profile().intrinsics
        color_1 = frames_1.get_color_frame()
        color_image_1 = np.asanyarray(color_1.get_data())
        # Initialize the cubemos api with a valid license key in default_license_dir()
        skeletrack = skeletontracker(cloud_tracking_api_key="")
        joint_confidence = 0.2
        # Create window for initialisation
        while True:
            # Create a pipeline object. This object configures the streaming camera and owns it's handle
            unaligned_frames_1 = pipeline_1.wait_for_frames()
            frames_1 = align_1.process(unaligned_frames_1)
            depth_1 = frames_1.get_depth_frame()
            color_1 = frames_1.get_color_frame()
            if not depth_1 or not color_1:
                continue
            color_image_1 = np.asanyarray(color_1.get_data())
            color_image_1 = cv2.cvtColor(color_image_1, cv2.COLOR_BGR2RGB)
            skeletons = skeletrack.track_skeletons(color_image_1)
            # print(skeletons)
            # render the skeletons on top of the acquired image and display it
            cm.render_result(skeletons, color_image_1, joint_confidence)
            
            skeletons_3D = render_ids_3d(
                color_image_1, skeletons, depth_1, depth_intrinsic_1, joint_confidence
            )
            dsize = (320, 240)
            color_image_1 = cv2.resize(color_image_1, dsize)
            ret, buffer = cv2.imencode('.jpg', color_image_1)
            color_image_1 = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + color_image_1 + b'\r\n')  # concat frame one by one and show result
    elif int(camera_id) == 1:
        config_2 = rs.config()
        config_2.enable_device('046122251324')
        config_2.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
        config_2.enable_stream(rs.stream.color, 848, 480, rs.format.rgb8, 30)
        colorizer = rs.colorizer()
        # Start the realsense pipeline
        pipeline_2 = rs.pipeline()
        pipeline_2.start(config_2)
        # Create align object to align depth frames to color frames
        align_2 = rs.align(rs.stream.color)
        for i in range(30):
            pipeline_2.wait_for_frames()
        # Get the intrinsics information for calculation of 3D point
        unaligned_frames_2 = pipeline_2.wait_for_frames()
        frames_2 = align_2.process(unaligned_frames_2)
        depth_2 = frames_2.get_depth_frame()
        depth_intrinsic_2 = depth_2.profile.as_video_stream_profile().intrinsics
        color_2 = frames_2.get_color_frame()
        color_image_2 = np.asanyarray(color_2.get_data())
        # Initialize the cubemos api with a valid license key in default_license_dir()
        skeletrack = skeletontracker(cloud_tracking_api_key="")
        joint_confidence = 0.2
        # Create window for initialisation
        print('hi')
        while True:
            # Create a pipeline object. This object configures the streaming camera and owns it's handle
            unaligned_frames_2 = pipeline_2.wait_for_frames()
            frames_2 = align_2.process(unaligned_frames_2)
            depth_2 = frames_2.get_depth_frame()
            color_2 = frames_2.get_color_frame()
            if not depth_2 or not color_2:
                continue
            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_2.get_data())
            dsize = (320, 240)
            # resize image
            depth_image = cv2.resize(depth_image, dsize)
            ret, b = cv2.imencode('.jpg', depth_image)
            depth_image = b.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + depth_image + b'\r\n')  # concat frame one by one and show result
#run show page
@app.route('/video_feed/<string:id>/', methods=["GET"])
def video_feed(id):
    print('id:',id)
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frames(id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
# dau tien vao se co cai nay
@app.route('/') #tao API
def home(): 
    """Video streaming home page."""
    return render_template('home.html')
@app.route('/show')
def show():
    return render_template('show.html')

@app.route('/loading')
def loading():
    # write into database
    # mycursor = mydb.cursor()
    # sql = "INSERT INTO loading (id,flag) VALUES (%s, %s)"
    # val = (None,1)
    # mycursor.execute(sql, val)
    # mydb.commit()
    return render_template('loading.html')

@app.route('/interaction')
def interaction():
    """Video streaming home page."""
    return render_template('interaction.html')

if __name__ == '__main__':
    try:
        app.run(debug=False)        # Configure depth and color streams of the intel realsense
    except Exception as ex:

        print('Exception occured: "{}"'.format(ex))

