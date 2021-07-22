from flask import Flask, render_template, Response
import cv2
import time
import mysql.connector
app = Flask(__name__)

mydb = mysql.connector.connect(
  host="localhost",
  user="byt",
  password="1231",
  database="interaction"
)

def gen_frames(camera_id):
    # cap = cv2.VideoCapture(0)
    # print('camera_id:',camera_id)
    if int(camera_id) == 0:
            # for cap in caps:
            # # Capture frame-by-frame
            # success, frame = cap.read()  # read the camera frame
            print('vao while 0')
            cap = cv2.VideoCapture(0)
            # frame = cv2.imread('hcm.png')
            while(cap.isOpened()):
                ret, frame = cap.read()
                if not ret:
                    pass
                # dsize
                dsize = (320, 240)
                # resize image
                frame = cv2.resize(frame, dsize)
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
    elif int(camera_id) == 1:
            # for cap in caps:
            # # Capture frame-by-frame
            # success, frame2 = cap.read()  # read the camera frame
            # frame2 = cv2.imread('hanoi.jpg')
            print('vao while 1')

            cap2 = cv2.VideoCapture(0)
            # frame = cv2.imread('hcm.png')
            while(cap2.isOpened()):
                ret, frame2 = cap2.read()
                if not ret:
                    pass
                 # dsize
                dsize = (320, 240)
                # resize image
                frame2 = cv2.resize(frame2, dsize)
                ret, buffer = cv2.imencode('.jpg', frame2)
                frame2 = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame2 + b'\r\n')  # concat frame one by one and show result
@app.route('/video_feed/<string:id>/', methods=["GET"])
def video_feed(id):
    print('id:',id)
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frames(id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/')
def home(): 
    """Video streaming home page."""
    return render_template('home.html')
@app.route('/show')
def show():
    return render_template('show.html')
@app.route('/loading')
def loading():
    #write into database
    mycursor = mydb.cursor()
    sql = "INSERT INTO size (id, g,h,v1,v2,v3) VALUES (%s, %s,%s, %s,%s, %s)"
    val = (None)
    mycursor.execute(sql, val)
    mydb.commit()

    return render_template('loading.html')
    
@app.route('/interaction')
def interaction():
    """Video streaming home page."""
    return render_template('interaction.html')

if __name__ == '__main__':
    app.run(debug=False)