import datetime
import os

import numpy.polynomial as poly
import cv2
import numpy as np
import lane_detection


def time_stamped_image_path(root):
    """ Create a file path for an image using a time stamp as a file name. """
    t_stamp = datetime.datetime.now().isoformat()
    t_stamp = t_stamp.replace(':', '-').replace('.', '-')
    return os.path.join(root, t_stamp) + '.png'


def draw_lane(lane, src):
    """ Draws a (numpy) polynomial into an image. """
    X, Y = reversed(lane.linspace())
    out = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
    for x, y in zip(X.tolist(), Y.tolist()):
        cv2.circle(out, (int(x), int(y)), 5, (0, 0, 255), -1)
    return out


def write_debug_image(root, src):
    if root is not None:
        cv2.imwrite(time_stamped_image_path(root), src)


class LaneTracker:
    """ A class for tracking lane lines in a video stream.

    TODO: Combine this class with the KalmanFilter class.
    """

    def __init__(self, input_size, output_size, m, threshold, d_set, n_windows, window_margin, min_n_pixel, y_step_max,
                 use_red_channel, blur_kernel_size, debug_images_path):
        """ Default initializer for the tracker class. """
        # TRACKED VALUES
        # ==============
        self.lanes = []
        self.e_d = 0.0
        self.e_alpha = 0.0
        self.right_x_base = None

        # TRACKING PARAMETERS
        # ===================
        self.input_size = input_size
        self.output_size = output_size
        self.m = m
        self.d_set = d_set
        self.n_windows = n_windows
        self.window_margin = window_margin
        self.min_n_pixel = min_n_pixel
        self.y_step_max = y_step_max
        self.threshold = threshold
        self.debug_images_path = debug_images_path
        self.use_red_channel = use_red_channel
        self.blur_kernel_size = blur_kernel_size

    def _run_pipeline(self, src):
        """
        Runs the tracking pipeline.

        Parameters
        ----------
        src : ndarray
            A binary image that is suitable for tracking.

        """
        image_bw_bird = cv2.warpPerspective(src, self.m, self.output_size, flags=cv2.WARP_INVERSE_MAP)

        # detect points on the lane lines and approximate with a polynomial
        rightx, righty, self.right_x_base = lane_detection.detect_points(image_bw_bird,
                                                                         self.n_windows,
                                                                         self.window_margin,
                                                                         self.min_n_pixel,
                                                                         self.y_step_max,
                                                                         self.right_x_base)
        self.lanes = [poly.Polynomial.fit(righty, rightx, 2)]

        # compute resulting positioning
        self.e_d, self.e_alpha = lane_detection.compute_positioning(image_bw_bird, self.lanes, self.d_set)
        out = draw_lane(self.lanes[0], image_bw_bird)
        write_debug_image(self.debug_images_path, out)

    def filter_lanes(self, frame):
        """
        Processes a camera image to produce a representation that the lane tracker is able to handle.

        (This is a very simple filtering approach based on a threshold for red/white.)
        TODO: Verify that the closing/smoothing steps are required.

        Parameters
        ----------
        frame : ndarray
            Input BGR-color-space image.

        Returns
        -------
        processed_frame : ndarray
            A binary image where everything except the lane-lines is set to zero.

        """
        src = cv2.resize(frame, (self.input_size[1], self.input_size[0]))

        if self.use_red_channel:
            # Compute and filter gradient of the image.
            src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            sobelx = cv2.Sobel(src, cv2.CV_64F, 1, 0, ksize=9)
            sobelx[sobelx > 0] = 0
            sobelx *= -1
            sobelx[sobelx < sobelx.max() * 0.15] = 0
            sobelx[sobelx > sobelx.max() * 0.15] = 255
            src = sobelx.astype(np.uint8)
        else:
            src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

            # use smoothing (normalized box filter) to remove noise from the image
            blurred = cv2.blur(src, self.blur_kernel_size)
            src = cv2.threshold(blurred, self.threshold, 255, cv2.THRESH_BINARY)[1]

        write_debug_image(self.debug_images_path, cv2.resize(frame, (self.input_size[1], self.input_size[0])))
        write_debug_image(self.debug_images_path, src)
        return src

    def handle_frame(self, frame):
        """ Updates the positioning values using the specified frame. """
        src = self.filter_lanes(frame)
        self._run_pipeline(src)


class KalmanFilter:
    """ A class for a Kalman-filter of the Ackermann-model. """

    def __init__(self, v, l_H, l, Ts):
        # PARAMETERS
        self.v = v
        self.l_H = l_H
        self.l = l
        self.Ts = Ts

        '''System configuration'''
        self.kalman = cv2.KalmanFilter(2, 2, 1)

        '''Initialization'''
        # Corrected error estimate covariance matrix P_k,+
        self.kalman.errorCovPost = np.eye(2)
        # Corrected estimate state x_k,+
        self.kalman.statePost = np.array([[40.], [0.]])
        self.set_up()

    def set_up(self):
        """ Update A & B matrixes according to given parameters. """
        # System matrix A
        self.kalman.transitionMatrix = np.array([[1., self.v * self.Ts], [0., 1.]])
        # Control matrix B
        self.kalman.controlMatrix = np.array(
            [[self.v * self.l_H * self.Ts / self.l + (self.v * self.Ts) ** 2 / (2 * self.l)],
             [self.v * self.Ts / self.l]])
        # Observation matrix C
        self.kalman.measurementMatrix = np.eye(2)
        # System noise covariance matrix Q
        self.kalman.processNoiseCov = np.array([[500, 0],
                                                [0, 0.5]])
        # Measurement noise covariance matrix R
        self.kalman.measurementNoiseCov = np.array([[100, 0],
                                                    [0, 0.0141]])
