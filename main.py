import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import roslib
import rospy
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import Image
from sensor_msgs.msg import Imu

from cv_bridge import CvBridge, CvBridgeError
import cv2.aruco as aruco
from scipy.spatial.transform import Rotation as R
import time
import tf

HEIGHT_CUT = 200
VERBOSE = True

aruco_dict = aruco.custom_dictionary(0, 7)
aruco_dict.bytesList = np.zeros(shape=(16, 7, 4), dtype=np.uint8)

ar = np.array([[1, 1, 0, 1, 1],
               [1, 1, 0, 1, 1],
               [1, 0, 1, 0, 1],
               [0, 0, 1, 1, 0],
               [1, 1, 1, 0, 1]], dtype=np.uint8)
aruco_dict.bytesList[1] = aruco.Dictionary_getByteListFromBits(
    np.pad(ar, pad_width=1, mode='constant', constant_values=0))

ar = np.array([[1, 1, 0, 1, 1],
               [1, 1, 0, 1, 1],
               [1, 0, 1, 0, 1],
               [1, 0, 1, 1, 0],
               [1, 0, 1, 1, 0]], dtype=np.uint8)
aruco_dict.bytesList[2] = aruco.Dictionary_getByteListFromBits(
    np.pad(ar, pad_width=1, mode='constant', constant_values=0))

ar = np.array([[1, 1, 0, 1, 1],
               [1, 1, 0, 1, 1],
               [1, 0, 1, 0, 1],
               [0, 1, 1, 1, 1],
               [1, 0, 1, 0, 0]], dtype=np.uint8)
aruco_dict.bytesList[3] = aruco.Dictionary_getByteListFromBits(
    np.pad(ar, pad_width=1, mode='constant', constant_values=0))

ar = np.array([[1, 1, 0, 1, 1],
               [1, 1, 0, 1, 1],
               [1, 0, 1, 0, 1],
               [0, 1, 1, 1, 0],
               [0, 1, 1, 1, 0]], dtype=np.uint8)
aruco_dict.bytesList[4] = aruco.Dictionary_getByteListFromBits(
    np.pad(ar, pad_width=1, mode='constant', constant_values=0))

ar = np.array([[1, 1, 0, 1, 1],
               [1, 1, 0, 1, 1],
               [1, 0, 1, 0, 1],
               [1, 0, 1, 1, 1],
               [0, 1, 1, 0, 0]], dtype=np.uint8)
aruco_dict.bytesList[5] = aruco.Dictionary_getByteListFromBits(
    np.pad(ar, pad_width=1, mode='constant', constant_values=0))

ar = np.array([[1, 1, 0, 1, 1],
               [1, 1, 0, 1, 1],
               [1, 0, 1, 0, 1],
               [0, 0, 1, 1, 1],
               [0, 0, 1, 1, 1]], dtype=np.uint8)
aruco_dict.bytesList[6] = aruco.Dictionary_getByteListFromBits(
    np.pad(ar, pad_width=1, mode='constant', constant_values=0))

ar = np.array([[1, 1, 0, 1, 1],
               [1, 1, 0, 1, 1],
               [1, 0, 1, 0, 1],
               [1, 1, 1, 1, 0],
               [0, 0, 1, 0, 1]], dtype=np.uint8)
aruco_dict.bytesList[7] = aruco.Dictionary_getByteListFromBits(
    np.pad(ar, pad_width=1, mode='constant', constant_values=0))

ar = np.array([[1, 1, 0, 1, 1],
               [1, 1, 0, 1, 1],
               [1, 0, 1, 0, 1],
               [0, 0, 1, 0, 1],
               [1, 1, 1, 1, 0]], dtype=np.uint8)
aruco_dict.bytesList[8] = aruco.Dictionary_getByteListFromBits(
    np.pad(ar, pad_width=1, mode='constant', constant_values=0))

ar = np.array([[1, 1, 0, 1, 1],
               [1, 1, 0, 1, 1],
               [1, 0, 1, 0, 1],
               [1, 1, 1, 0, 0],
               [1, 1, 1, 0, 0]], dtype=np.uint8)
aruco_dict.bytesList[9] = aruco.Dictionary_getByteListFromBits(
    np.pad(ar, pad_width=1, mode='constant', constant_values=0))

ar = np.array([[1, 1, 0, 1, 1],
               [1, 1, 0, 1, 1],
               [1, 0, 1, 0, 1],
               [0, 1, 1, 0, 0],
               [1, 0, 1, 1, 1]], dtype=np.uint8)
aruco_dict.bytesList[10] = aruco.Dictionary_getByteListFromBits(
    np.pad(ar, pad_width=1, mode='constant', constant_values=0))

ar = np.array([[1, 1, 0, 1, 1],
               [1, 1, 0, 1, 1],
               [1, 0, 1, 0, 1],
               [1, 0, 1, 0, 1],
               [1, 0, 1, 0, 1]], dtype=np.uint8)
aruco_dict.bytesList[11] = aruco.Dictionary_getByteListFromBits(
    np.pad(ar, pad_width=1, mode='constant', constant_values=0))

ar = np.array([[1, 1, 0, 1, 1],
               [1, 1, 0, 1, 1],
               [1, 0, 1, 0, 1],
               [1, 0, 1, 0, 0],
               [0, 1, 1, 1, 1]], dtype=np.uint8)
aruco_dict.bytesList[12] = aruco.Dictionary_getByteListFromBits(
    np.pad(ar, pad_width=1, mode='constant', constant_values=0))

ar = np.array([[1, 1, 0, 1, 1],
               [1, 1, 0, 1, 1],
               [1, 0, 1, 0, 1],
               [0, 1, 1, 0, 1],
               [0, 1, 1, 0, 1]], dtype=np.uint8)
aruco_dict.bytesList[13] = aruco.Dictionary_getByteListFromBits(
    np.pad(ar, pad_width=1, mode='constant', constant_values=0))

ar = np.array([[1, 1, 0, 1, 1],
               [1, 1, 0, 1, 1],
               [1, 0, 1, 0, 1],
               [1, 1, 1, 0, 1],
               [0, 0, 1, 1, 0]], dtype=np.uint8)
aruco_dict.bytesList[14] = aruco.Dictionary_getByteListFromBits(
    np.pad(ar, pad_width=1, mode='constant', constant_values=0))

ar = np.array([[1, 1, 0, 1, 1],
               [1, 1, 0, 1, 1],
               [1, 0, 1, 0, 1],
               [0, 0, 1, 0, 0],
               [0, 0, 1, 0, 0]], dtype=np.uint8)
aruco_dict.bytesList[15] = aruco.Dictionary_getByteListFromBits(
    np.pad(ar, pad_width=1, mode='constant', constant_values=0))

print(aruco_dict)


def find_stones(img):
    blur = cv2.blur(img, (8, 8))
    blur = cv2.blur(blur, (5, 5))
    blur = cv2.blur(blur, (5, 5))
    blur = cv2.blur(blur, (5, 5))
    blur = cv2.GaussianBlur(blur, (5, 5), 0)
    blur = cv2.GaussianBlur(blur, (5, 5), 0)
    blur = cv2.GaussianBlur(blur, (5, 5), 0)

    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    bright = hsv[:, :, 2].copy()
    hsv[:, :, 0] = 0
    hsv[:, :, 2] = 0

    new_gray = hsv[:, :, 1]

    kernel = np.ones((20, 20), np.uint8)
    new_gray = cv2.morphologyEx(new_gray, cv2.MORPH_CLOSE, kernel)

    new_gray = cv2.GaussianBlur(new_gray, (5, 5), 0)

    new_gray *= int(255 / (new_gray.max() if new_gray.max() != 0 else 0.0000001))

    cut_img_gray = new_gray[HEIGHT_CUT:, :]
    stone_min_t = np.mean(cut_img_gray)
    STONE_MIN_THRESH = stone_min_t * 2
    ret, thresh = cv2.threshold(new_gray, STONE_MIN_THRESH, 255, cv2.THRESH_BINARY_INV)

    bright_kernel = np.ones((8, 8), np.uint8)
    bright_closing = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, bright_kernel)

    bright_closing *= int(255 / (bright_closing.max() if bright_closing.max() != 0 else 0.0000001))

    SHADOW_MIN_THRESH = 100
    ret, thresh_bright = cv2.threshold(bright_closing, SHADOW_MIN_THRESH, 255, cv2.THRESH_BINARY)

    resulting_thresh = cv2.bitwise_and((cv2.bitwise_not(thresh) + cv2.bitwise_not(thresh_bright)), thresh_bright)

    resulting_thresh = resulting_thresh.copy()[HEIGHT_CUT:, :]
    img_contours, _ = cv2.findContours(resulting_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_with_stone_contours = img.copy()[HEIGHT_CUT:, :]
    STONE_AREA_MIN_SIZE = 1000  # TODO calculate by Y
    stone_contours = []
    for cnt in img_contours:
        if (cv2.contourArea(cnt) > STONE_AREA_MIN_SIZE):
            print(cv2.contourArea(cnt))
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                M["m00"] == 0.00000001
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            print(cY)
            cv2.circle(img_with_stone_contours, (cX, cY), 5, (0, 0, 255), -1)
            stone_contours.append(cnt)
    if len(stone_contours) > 0:
        cv2.drawContours(img_with_stone_contours, stone_contours, 5, (255, 0, 0))
    return img_with_stone_contours


class image_feature:
    def __init__(self):

        self.image_pub = rospy.Publisher("/output/image_raw/compressed", CompressedImage)
        self.bridge = CvBridge()
        self.vel = 0
        self.last_time = time.time()
        self.subscriber = rospy.Subscriber("/zed2/left_raw/image_raw_color", Image, self.callback,
                                           queue_size=10)
        self.start_ryskanie = None
        self.imu = None
        self.last_dist = 0
        self.subscriber = rospy.Subscriber("/wheel_odom", TwistStamped, self.callback_odom,
                                           queue_size=10)

        self.subscriber = rospy.Subscriber("/zed2/imu/data", Imu, self.callback_imu,
                                           queue_size=10)

        if VERBOSE:
            print("subscribed to /camera/image/compressed")

    def callback_odom(self, msg):
        self.vel = msg.twist.linear.x

    def callback_imu(self, msg):
        self.imu = msg

    def callback(self, ros_data):
        '''Callback function of subscribed topic.
        Here images get converted and features detected'''

        img = self.bridge.imgmsg_to_cv2(ros_data)
        if VERBOSE:
            pass
            # print('received image of type: "%s"' % ros_data.format)
        #### direct conversion to CV2 ####
        image_np = np.array(img[:, :, :3], np.uint8)
        # image_np = cv2.imdecode(np_arr, cv2.CV_LOAD_IMAGE_COLOR)
        # image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # OpenCV >= 3.0:
        print(image_np.shape)

        res_img = self.find_ars(image_np)

        print(res_img.shape)
        cv2.imshow("found stones", res_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            pass

        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', image_np)[1]).tostring()
        # Publish new image
        self.image_pub.publish(msg)

    def find_ars(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        matrix_coefficients = np.array([[500., 0, 0], [0, 500, 0], [0, 0, 1]])
        distortion_coefficients = np.array(
            [-0.04369359835982323, 0.014616499654948711, -0.006573319900780916, -0.00021690200082957745,
             0.0000843286034069024])

        parameters = aruco.DetectorParameters_create()
        corners, ids, rejected_img_points = aruco.detectMarkers(gray, aruco_dict, cameraMatrix=matrix_coefficients,
                                                                distCoeff=distortion_coefficients,
                                                                parameters=parameters)
        aruco.drawDetectedMarkers(img, corners)
        print('corners', corners)

        font = cv2.FONT_HERSHEY_SIMPLEX

        fontScale = 0.8
        color = (0, 0, 255)
        thickness = 2

        if len(corners) > 0:
            org = (corners[0][0][0][0], corners[0][0][0][1])

            corners_ = corners[0][0]
            area = cv2.contourArea(corners_)
            print("CORNERS")
            print(corners_)

            fuckalnaya_huinya_v_metrah = 0.002
            coef_px2_to_m2 = (1 / (6.25 * 10 ** 10))

            rvecs, tvecs, _objPoints = aruco.estimatePoseSingleMarkers(corners, 0.05, matrix_coefficients,
                                                                       distortion_coefficients)

            print("ROTATION HUINYA")

            r = R.from_rotvec(rvecs[0])
            print(r.as_matrix())
            normalnaya_huinya = (r.as_matrix() * np.array([1, 0, 0]).transpose())[0][:, 0]
            area_real = 0.135 * 0.135

            print(normalnaya_huinya)

            rasstoyanie = fuckalnaya_huinya_v_metrah * np.sqrt(
                area_real * np.dot(np.array([1, 0, 0]), normalnaya_huinya) / (
                        np.linalg.norm(np.array([1, 0, 0])) * np.linalg.norm(
                    normalnaya_huinya) * area * coef_px2_to_m2))

            print(rasstoyanie)
            cv2.putText(img, str(rasstoyanie), org, font,
                        fontScale, color, thickness, cv2.LINE_AA)

            # aruco.drawAxis(img, matrix_coefficients, distortion_coefficients, rvecs[0], tvecs[0], 0.1)

            cur_time = time.time()
            dt = cur_time - self.last_time
            ddist = self.last_dist - rasstoyanie

            print("ODOM : ", self.vel)
            print("DDIST: ", ddist / dt)

            self.last_dist = rasstoyanie
            self.last_time = cur_time

            print('tvecs', tvecs)

        if self.imu:

            (r, p, y) = tf.transformations.euler_from_quaternion(
                [self.imu.orientation.x, self.imu.orientation.y, self.imu.orientation.z,
                 self.imu.orientation.w])

            if not self.start_ryskanie:
                self.start_ryskanie = y
            else:
                cv2.putText(img, f"kurs     {round((y- self.start_ryskanie) / np.pi * 180)}", (50, 600), font,
                            fontScale, color, thickness, cv2.LINE_AA)

            cv2.putText(img, f"kren.     {round(r / np.pi * 180)}", (50, 50), font,
                        fontScale, color, thickness, cv2.LINE_AA)
            cv2.putText(img, f"tangazh  {round(p / np.pi * 180)}", (50, 100), font,
            fontScale, color, thickness, cv2.LINE_AA)
            cv2.putText(img, f"ryskanie  {round(y / np.pi * 180)}", (50, 150), font,
            fontScale, color, thickness, cv2.LINE_AA)

        return img


def main():
    '''Initializes and cleanup ros node'''
    ic = image_feature()
    rospy.init_node('stone_detection', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS Image feature detector module")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
