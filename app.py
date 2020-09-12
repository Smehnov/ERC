import numpy as np
import cv2
import rospy
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import TwistStamped
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from sensor_msgs.msg import Imu
from std_msgs.msg import String
from geometry_msgs.msg import TwistStamped
from cv_bridge import CvBridge, CvBridgeError
import time
from ars import aruco_dict
import cv2.aruco as aruco

# ROS rate
RATE = 25  # Hz

# Angular movement
EPSILON_RAD = 0.05  # rad !!!
ANG_SAT = 0.6  # rad/s
K_a_P = 0.5
K_a_I = 0.15

# Linear movement
EPSILON_LIN = 0.06  # m
LIN_SAT = 0.25  # m/s
K_l_P = 0.2


class DummyController:
    def __init__(self):
        self.img = None
        self.last_time_aruco_saved = time.time()
        self.bridge = CvBridge()
        self.vel = 0
        self.last_time = time.time()
        self.start_ryskanie = None
        self.kurs = None
        self.deltaKurs = None
        self.imu = None
        self.last_dist = 0
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.subscriber = rospy.Subscriber("/zed2/imu/data", Imu, self.callback_imu,
                                           queue_size=10)

        self.image_subscriber = rospy.Subscriber("/zed2/left_raw/image_raw_color", Image, self.callback_img,
                                                 queue_size=10)

        self.command_subscriber = rospy.Subscriber("/command", String, self.callback_command,
                                                   queue_size=10)
        self.odom_vx = 0
        self.odom_subscriber = rospy.Subscriber("/wheel_odom", TwistStamped, self.callback_odom,
                                                queue_size=10)
        # self.rate = rospy.Rate(25)

    def clear_commands(self):
        pass

    def take_photo(self):
        print("PHOTO TAKEN")
        try:
            cv2.imwrite("." + str(time.time()) + "taken.png", self.img)
        except:
            print("ERROR WHILE TAKING PHOTO")

    def callback_command(self, msg):
        command = str(msg.data)
        self.clear_commands()

        if command.startswith('turn'):
            angle = int(command.split(' ')[1])
            self.set_kurs(angle)
        if command.startswith('go'):
            dist = float(command.split(' ')[1])
            self.move_forward(dist)

        if command.startswith('photo'):
            self.take_photo()

    def callback_odom(self, msg):
        self.odom_vx = msg.twist.linear.x

    def callback_img(self, ros_data):
        '''Callback function of subscribed topic.
        Here images get converted and features detected'''

        img = self.bridge.imgmsg_to_cv2(ros_data)
        image_np = np.array(img[:, :, :3], np.uint8)
        # print(image_np.shape)
        self.img = image_np

        self.find_ars(image_np)

    def set_speed(self, v=0, w=0):
        twist = Twist()

        twist.linear.x = v
        twist.angular.z = w
        self.cmd_pub.publish(twist)

    def set_kurs(self, alpha):
        # alpha in deg!!!!
        self.kurs = (float(alpha) / 180) * np.pi
        self.find_kurs()

    def callback_imu(self, msg):
        self.imu = msg
        (r, p, y) = tf.transformations.euler_from_quaternion(
            [self.imu.orientation.x, self.imu.orientation.y, self.imu.orientation.z,
             self.imu.orientation.w])

        # print("kren.    : ", round(r / np.pi * 180))
        # print("tangazh  : ", round(p / np.pi * 180))
        # print("ryskanie : ", round(y / np.pi * 180))
        if self.kurs is not None:
            self.deltaKurs = self.kurs - y
            if abs(self.deltaKurs) > np.pi:
                self.deltaKurs = self.deltaKurs - np.sign(self.deltaKurs) * (2 * np.pi)

            # print("self.kurs : ",  round(self.kurs / np.pi * 180))
            #
            # print("del kurs : ", round(self.deltaKurs / np.pi * 180))

    def find_kurs(self, eps=EPSILON_RAD):

        e_i = 0
        while self.deltaKurs and abs(self.deltaKurs) > eps:
            omega = K_a_P * self.deltaKurs + K_a_I * e_i

            e_i += self.deltaKurs * (1.0 / RATE)

            print("--- e_i : ", e_i, " ---")

            if omega > ANG_SAT:
                self.set_speed(0, ANG_SAT)
            elif omega < -ANG_SAT:
                self.set_speed(0, -ANG_SAT)
            else:
                self.set_speed(0, omega)
            rospy.Rate(RATE).sleep()

        self.set_speed(0, 0)
        rospy.Rate(RATE).sleep()

    def move_forward(self, d=0.0, eps=EPSILON_LIN):
        x = 0.0
        x_target = x + d

        err = x_target - x
        while abs(err) > eps:
            err = x_target - x
            x += self.odom_vx * (1.0 / RATE)

            print("lin err : ", err)

            v = err * K_l_P
            print("lin vel : ", v)
            if v > LIN_SAT:
                self.set_speed(LIN_SAT, 0)
            elif v < -LIN_SAT:
                self.set_speed(-LIN_SAT, 0)
            else:
                self.set_speed(LIN_SAT, 0)
            rospy.Rate(RATE).sleep()

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
        if len(ids) > 0 and self.last_time_aruco_saved - time.time() > 5:
            self.last_time_aruco_saved = time.time()
            cv2.imwrite(str(time.time()) + "_" + str(ids[0]) + ".png", img)


def main():
    '''Initializes and cleanup ros node'''
    rospy.init_node('rover', anonymous=True)
    controller = DummyController()

    rate = rospy.Rate(RATE)
    while not rospy.is_shutdown():
        try:

            rate.sleep()

        except KeyboardInterrupt:
            print("Shutting down ROS Image feature detector module")


if __name__ == '__main__':
    main()
