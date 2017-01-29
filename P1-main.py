#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

import math


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


def color_selec(image):
    # Grab the x and y size and make a copy of the image
    ysize = image.shape[0]
    xsize = image.shape[1]
    color_select = np.copy(image)

    # Define color selection criteria
    red_threshold = 190
    green_threshold = 190
    blue_threshold = 0  # to select yellow lanes
    ######

    rgb_threshold = [red_threshold, green_threshold, blue_threshold]

    # Do a boolean or with the "|" character to identify
    # pixels below the thresholds
    thresholds = (image[:, :, 0] < rgb_threshold[0]) \
                 | (image[:, :, 1] < rgb_threshold[1]) \
                 | (image[:, :, 2] < rgb_threshold[2])
    color_select[thresholds] = [0, 0, 0]
    return color_select


def extract_lines(img, prc_img):
    # Lines extraction
    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 20  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 20  # minimum number of pixels making up a line
    max_line_gap = 200  # maximum gap in pixels between connectable line segments

    lines = cv2.HoughLinesP(prc_img, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
    line_img = np.copy(img)
    # if lines.any():
    #    draw_lines(line_img, lines)

    return (line_img, lines)


def extrapolate(lines, img_height):
    if len(lines) != 0:
        """
        Classify lines into 2 categories: left or right and ignore the other lines not fitting
        With the 2 arrays of right/left lines, average the slope & extrapolate
        """
        RIGHT_LANE_MIN_SLOPE = 0.5
        RIGHT_LANE_MAX_SLOPE = 0.9
        LEFT_LANE_MAX_SLOPE = - 0.5
        LEFT_LANE_MIN_SLOPE = - 0.9
        left_lines = []
        # Find the bottom start point for left
        Y_IMG_LENGTH = img_height
        left_min_x = Y_IMG_LENGTH  # arbitraty value to init max pixel value
        left_max_y = 0
        left_min_y = Y_IMG_LENGTH
        # Find the bottom left point for right
        right_max_x = 0
        right_max_y = 0
        right_min_y = Y_IMG_LENGTH
        # Average Slopes
        avg_left_slp = 0
        avg_right_slp = 0
        right_lines = []
        for line in lines:
            for x1, y1, x2, y2 in line:
                slp = (float(y2) - y1) / (x2 - x1)
                # Classify into left or right and compute averaged slope
                if slp > LEFT_LANE_MIN_SLOPE and slp < LEFT_LANE_MAX_SLOPE:
                    left_lines.append([x1, y1, x2, y2, slp])
                    avg_left_slp += slp
                    # Find the most bottom left point of all the lines
                    left_min_x = min(left_min_x, x1, x2)
                    left_max_y = max(left_max_y, y1, y2)
                    left_min_y = min(left_min_y, y1, y2)
                if slp > RIGHT_LANE_MIN_SLOPE and slp < RIGHT_LANE_MAX_SLOPE:
                    right_lines.append([x1, y1, x2, y2, slp])
                    avg_right_slp += slp
                    # Find the most bottom right point of all the lines
                    right_max_x = max(right_max_x, x1, x2)
                    right_max_y = max(right_max_y, y1, y2)
                    right_min_y = min(right_min_y, y1, y2)

        if right_lines and left_lines:
            avg_right_slp = avg_right_slp / len(right_lines)
            avg_left_slp = avg_left_slp / len(left_lines)
        if avg_left_slp and avg_right_slp:
            # Find the missing parameter to define y = ax + b
            b1 = left_max_y - left_min_x * avg_left_slp
            b2 = right_max_y - right_max_x * avg_right_slp
            # Extrapolate distance is at least 200 px from bottom but can be more if a line was found far enough
            extrapol_dist_y = min(Y_IMG_LENGTH - 200, left_min_y, right_min_y)
            line_start_x1 = int((Y_IMG_LENGTH - b1) / avg_left_slp)
            line_start_x2 = int((Y_IMG_LENGTH - b2) / avg_right_slp)
            line1 = [line_start_x1, Y_IMG_LENGTH, int((extrapol_dist_y - b1) / avg_left_slp), extrapol_dist_y]
            line2 = [line_start_x2, Y_IMG_LENGTH, int((extrapol_dist_y - b2) / avg_right_slp), extrapol_dist_y]
        else:
            line1 = [0, 0, 0, 0]
            line2 = [0, 0, 0, 0]
    else:
        line1 = [0, 0, 0, 0]
        line2 = [0, 0, 0, 0]
    return line1, line2


def process_pipeline(image):
    # Agressive Select by color
    prc_img = color_selec(image)
    # Back to gray scale for canny
    prc_img = grayscale(prc_img)
    # Keep only interesting part
    vertices = np.array([[(120, 539), (900, 539), [475, 300], [130, 539]]], dtype=np.int32)
    prc_img = region_of_interest(prc_img, vertices)
    # Reduce high variations like textures or noise
    prc_img = gaussian_blur(prc_img, 7)
    # Contour extraction
    prc_img = canny(prc_img, 50, 150)
    # Lines extraction
    [line_image, lines] = extract_lines(image, prc_img)
    [line1, line2] = extrapolate(lines, image.shape[0])
    # Copy image to overlay result on it without modifying original
    # final_img = np.copy(image)
    # Overlay computed lines
    cv2.line(line_image, (line1[0], line1[1]), (line1[2], line1[3]), color=[0, 255, 0], thickness=5)
    cv2.line(line_image, (line2[0], line2[1]), (line2[2], line2[3]), color=[0, 255, 0], thickness=5)

    return line_image

## MAIN

#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')
#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
#plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')
#plt.show()


import os


def run_images():
    global image
    direc = "test_images/"
    images = os.listdir(direc)
    i = 0
    for idx, file in enumerate(images):
        i += 1
        fig = plt.figure(i)
        if ".jpg" in file:
            image = mpimg.imread(direc + file)
            plt.imshow(image)
            i += 1
            fig = plt.figure(i)
            plt.imshow(process_pipeline(image))
            print("processing done")
    plt.show(block=True)


# then save them to the test_images directory.
#run_images()
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # you should return the final output (image with lines are drawn on lanes)
    return process_pipeline(image)
# white_output = 'white.mp4'
# clip1 = VideoFileClip("solidWhiteRight.mp4")
# white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
# white_clip.write_videofile(white_output, audio=False)
#
# yellow_output = 'yellow.mp4'
# clip2 = VideoFileClip('solidYellowLeft.mp4')
# yellow_clip = clip2.fl_image(process_image)
# yellow_clip.write_videofile(yellow_output, audio=False)

challenge_output = 'extra.mp4'
clip2 = VideoFileClip('challenge.mp4')
challenge_clip = clip2.fl_image(process_image)
challenge_clip.write_videofile(challenge_output, audio=False)