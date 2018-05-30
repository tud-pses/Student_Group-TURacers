import numpy as np
import cv2


def compute_positioning(src, lanes, d_set=40):
    """
    Computes the car's position within a lane.

    Parameters
    ----------
    lanes
        The detected lanes in the image.
    src
        The source image
    d_set
        Set-point distance to the right lane.

    Returns
    -------
    e_d : double
        Distance between the car's center and the lane center ([e_d] ~ m).
    e_alpha : double
        The angular difference (in radians) between the heading of the car and lane orientation at the point closest to
        the car.

    """
    if len(lanes) != 1:
        raise ValueError('Computing a cars position is currently only supported for one lane.')
    p = lanes[0]

    height, width = src.shape
    p_der = p.deriv()
    x0 = height
    y0 = p(x0)

    # position error
    d = np.abs(width / 2 - y0)
    e_d = d_set - d

    # angular error
    e_alpha = np.arctan(p_der(x0))
    return e_d, e_alpha


def detect_points(binary_warped, n_windows=50, window_margin=15, min_n_pixel=20, y_step_max=100,
                  right_x_base=None, return_image=False):
    """
    Finds points that are likely to be located on the lane lines in an image (currently tries to find the right lane).

    Source:
        https://github.com/georgesung/advanced_lane_detection

    Parameters
    ----------
    return_image
        If True the function will return an image that visualizes the windowing process.
    binary_warped
        A binary image showing a birds eye view of the lanes.
    y_step_max
        The maximum distance between points in the y direction.
    min_n_pixel
        Minimum number of pixel inside a window to recenter.
    window_margin
        The windows +/- margin.
    n_windows
        Number of windows.

    """
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[-20:, :], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = (np.dstack((binary_warped, binary_warped, binary_warped)) * 255).astype('uint8')
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[15:midpoint]) + 15

    if right_x_base is None:
        rightx_base = np.argmax(histogram[midpoint:-15]) + midpoint

        # Find the starting point for the right lane
        boundary = midpoint
        while histogram[rightx_base] == np.mean(histogram[boundary:-15]):
            boundary -= 40
            rightx_base = np.argmax(histogram[boundary:-15]) + boundary
            if boundary < 0:
                raise ValueError('Lane detection failed: Unable to find a base point.')
    else:
        # search around last found base position
        b = int(1.1 * window_margin)
        boundary = right_x_base - b
        rightx_base = np.argmax(histogram[boundary:boundary+2*b]) + boundary

    # Choose the number of sliding windows
    nwindows = n_windows
    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = window_margin
    # Set minimum number of pixels found to recenter window
    minpix = min_n_pixel
    # Create empty lists to receive left and right lane pixel indices
    # left_lane_inds = []
    right_lane_inds = []

    # Configuration
    d_max = y_step_max
    # lefty_count = 1
    righty_count = 1

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        # good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
        #         nonzerox < win_xleft_high)).nonzero()[0]

        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
                nonzerox < win_xright_high)).nonzero()[0]

        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
            righty_current = np.int(np.mean(nonzeroy[good_right_inds]))
            # initialize righty_latest
            if righty_count == 1:
                righty_latest = righty_current
                righty_count += 1
            # When y of two neighbor points are far from each other, stop the loop
            dr = np.abs(righty_current - righty_latest)
            if dr > d_max:
                break

            # Update righy_latest
            righty_latest = righty_current
            # Append these indices to the lists
            right_lane_inds.append(good_right_inds)

    # Concatenate the arrays of indices
    # left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    # leftx = nonzerox[left_lane_inds]
    # lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    if return_image:
        return rightx, righty, out_img
    else:
        return rightx, righty, rightx_base

