#!/usr/bin/env python

import numpy as np
import datetime
import time
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Int16

from lane_detection.lane_tracker import LaneTracker, KalmanFilter

NODE_NAME = 'lane_keeping_node'
HALL_DT_TOPIC_NAME = '/uc_bridge/hallDt'
KINECT_IMAGES_TOPIC_NAME = '/kinect2/qhd/image_color'
SET_MOTOR_TOPIC_NAME = '/uc_bridge/set_motor_level_msg'
SET_STEERING_TOPIC_NAME = '/uc_bridge/set_steering_level_msg'

# =================================================================================
# Parameters for driving.
# TODO: Use ROS-Parameter-Server for setting parameters.
# =================================================================================
input_size = (135, 240)
output_size = (250, 100)
m = np.array([[5.00000000e+02, -1.18833504e+02, -2.00220859e+04],
              [0.00000000e+00, 2.70307922e+00, 3.41485703e+04],
              [0.00000000e+00, -9.90279198e-01, 3.53982605e+02]], dtype=np.float32)
d_set = 30
threshold = 220
n_windows = 20
window_margin = 15
min_n_pixel = 5
y_step_max = 250
steering_out_bounds = (-900, 900)
motor_level_bounds = [230, 280]
T_motor_lp = 2.
alpha_max = 0.12
k_stanley = 0.14
v_stanley_f = 1. / 200
radians_to_cnts = 1600 / np.pi
blur_kernel_size = (6, 6)
use_red_channel = True
l = 0.26
l_H = 0.13
Ts = 1. / 15
# ================================================================================


# setting this to a valid directory will cause the lane tracker to write debug images
DEBUG_IMAGES_PATH = None  # '/home/pses/lane_dct_debug3'


def _warn_if_processing_slow(dt):
    if 1 / dt < 15:
        rospy.logwarn('Processing rate (1/({} s) likely slower then kinect frame rate.'.format(dt))


class CarController:
    """ A class that handles driving within a lane. """

    def __init__(self):
        self.kinect_images_sub = rospy.Subscriber(KINECT_IMAGES_TOPIC_NAME, Image, self.on_frame)
        self.set_motor_level_pub = rospy.Publisher(SET_MOTOR_TOPIC_NAME, Int16, queue_size=1)
        self.set_steering_level_pub = rospy.Publisher(SET_STEERING_TOPIC_NAME, Int16, queue_size=1)
        self.bridge = CvBridge()

        self.lane_tracker = LaneTracker(input_size, output_size, m, threshold, d_set, n_windows, window_margin,
                                        min_n_pixel, y_step_max, use_red_channel, blur_kernel_size, DEBUG_IMAGES_PATH)

        # initialize the kalman filter
        self.kalman_filter = KalmanFilter(10.0, l_H, l, Ts)
        self.state_pre = np.zeros((2, 1))
        self.measurement = np.zeros((2, 1))
        self.state = np.zeros((2, 1))
        self.control = 0.0
        self.filtering()

        self.motor_level = 0
        self.motor_level_previous = motor_level_bounds[0]
        self.num_lane_detection_failed = 0

    def filtering(self):
        """ Kalman-filter predict and update step. """
        self.state_pre = self.kalman_filter.kalman.predict(np.array([self.control]))

        self.measurement = np.array([[d_set - self.lane_tracker.e_d], [self.lane_tracker.e_alpha]])
        self.state = self.kalman_filter.kalman.correct(self.measurement)

    def _pub_steering_level_stanley(self):
        """ Publish steering level using the stanley controller. """
        # Compute error from the last state of Kalman-filter
        e_d = d_set - self.kalman_filter.kalman.statePost[0][0]
        e_alpha = self.kalman_filter.kalman.statePost[1][0]

        # compute stanley control feedback
        v_stanley = self.motor_level * v_stanley_f
        phi_L = e_alpha + np.arctan(k_stanley * e_d / v_stanley)
        self.control = np.tan(phi_L)
        steering_out = radians_to_cnts * phi_L
        steering_out = -1 * max(steering_out_bounds[0], min(steering_out_bounds[1], steering_out))  # clamp

        self.set_steering_level_pub.publish(steering_out)

    def _pub_motor_level(self):
        e_alpha = np.abs(self.kalman_filter.kalman.statePost[1][0])

        # compute motor level as function of relative heading
        dx = - (motor_level_bounds[1] - motor_level_bounds[0]) / alpha_max
        m_level = np.abs(dx * e_alpha + motor_level_bounds[1])
        m_level = max(motor_level_bounds[0], min(motor_level_bounds[1], m_level))  # clamp

        # low pass filter motor level
        m_level_filtered = (1 / (T_motor_lp/Ts + 1)) * (m_level + (T_motor_lp/Ts)*self.motor_level_previous)
        self.motor_level_previous = m_level_filtered

        self.motor_level = m_level_filtered
        self.set_motor_level_pub.publish(self.motor_level)
        self.kalman_filter.v = self.motor_level * v_stanley_f
        self.kalman_filter.set_up()

    def on_frame(self, msg):
        """ Callback that handles a new frame from the kinect camera. """
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            t0 = time.time()

            # run detection and filtering pipeline
            self.lane_tracker.handle_frame(frame)
            self.filtering()

            # publish steering and motor levels
            self._pub_steering_level_stanley()
            self._pub_motor_level()

            _warn_if_processing_slow(time.time() - t0)
            self.num_lane_detection_failed = 0
        except CvBridgeError as _:
            # converting ros image to open-cv image failed
            # TODO: This might need some error handling ...
            pass
        except ValueError as e:
            # error likely due to missing / not-detected lane markings
            self.num_lane_detection_failed += 1

            # steer right on first detection failure
            # TODO: Find a way to remove this
            if self.num_lane_detection_failed <= 2:
                self.set_steering_level_pub.publish(steering_out_bounds[1])

            if self.num_lane_detection_failed > 2:
                rospy.logwarn('Lane detection failed for {} frames.'.format(self.num_lane_detection_failed))

                # reset the tracked right lane base position after multiple failed detection
                self.lane_tracker.right_x_base = None
                self._pub_steering_level_stanley()

            self.state_pre = self.kalman_filter.kalman.predict(np.array([self.control]))
            self._pub_motor_level()


def main():
    rospy.init_node(NODE_NAME, anonymous=True)
    ctrl = CarController()

    # keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == '__main__':
    main()
