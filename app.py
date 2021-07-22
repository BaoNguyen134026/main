from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)


def gen_frames(camera_id):
    cap = cv2.VideoCapture(0)
    print('camera_id:',camera_id)
    if int(camera_id) == 0:
            # for cap in caps:
            # # Capture frame-by-frame
            success, frame = cap.read()  # read the camera frame
            if not success:
                pass
            else:
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
    elif int(camera_id) == 1:
            # for cap in caps:
            # # Capture frame-by-frame
            success, frame2 = cap.read()  # read the camera frame
            if not success:
                pass
            else:
                gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
                ret, buffer = cv2.imencode('.jpg', gray)
                gray = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + gray + b'\r\n')  # concat frame one by one and show result

@app.route('/video_feed/<string:id>/', methods=["GET"])
def video_feed(id):
    print('id:',id)
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frames(id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
'''open page'''
@app.route('/')
def home(): 
    """Home page."""
    return render_template('home.html')
    
@app.route('/show')
def show():
    '''show page'''
    return render_template('show.html')
@app.route('/interaction')
def interaction():
    """interaction page"""
    return render_template('interaction.html')


'''main code'''
if __name__ == '__main__':
    try:
        app.run(debug=False)

    finally:
        pass