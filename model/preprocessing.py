from math import ceil, exp
import cv2 as cv
import numpy as np
from itertools import chain


def edge_detection(img0: np.ndarray, sigma: float):
    """
    Finds edge intensity and orientation in a greyscale image.

    :param img0: 2D numpy array containing a greyscale image
    :param sigma: standard deviation used for Gaussian Filter
    :return: 2D numpy array containing the edge magnitude image
    """
    edge_mag, edge_ori = edge_mag_ori(img0, sigma)
    post_nms = non_max_suppression(edge_mag, edge_ori)
    post_threshold = low_thresholding(post_nms, 15 / 255)

    return post_threshold


def edge_mag_ori(img: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Retrieves edge magnitude and orientation of greyscale image.

    :param img: 2D numpy array containing a greyscale image
    :param sigma: standard deviation used for Gaussian Filter
    :return: size 2 tuple containing edge magnitude and edge orientation
    """
    g_filter = make_gaussian_filter(sigma)
    g_blurred = img_filter(img, g_filter)
    edge_mag, edge_ori = sobel_filter(g_blurred)
    return edge_mag, edge_ori


def img_filter(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Applies filter on greyscale img using a kernel.

    :param img: 2D np.ndarray containing a greyscale image
    :param kernel: 2D np.ndarray containing filter
    :return: 2D np.ndarray containing input image after filter
    """
    filtered_img = np.zeros(img.shape)
    padded_img = reflect_pad(img, kernel.shape[0] // 2, kernel.shape[0] // 2)

    height, width = img.shape
    k_height, k_width = kernel.shape
    for h in range(height):
        for w in range(width):
            source_area = padded_img[h: h + k_height, w: w + k_width]
            convoluted_pixel = np.sum(np.multiply(kernel, source_area))     # Performs CELL multiplication - not matrix

            filtered_img[h][w] = convoluted_pixel

    return filtered_img


def make_gaussian_filter(sigma: float) -> np.ndarray:
    """
    Forms a Gaussian Filter based on sigma value.

    :param sigma: standard deviation of kernel; also used for determining kernel size
    :return: 2D numpy array containing kernel
    """
    half_size = ceil(3 * sigma)
    size = 2 * half_size + 1

    g_filter = np.zeros((size, size), np.float64)

    # By using this range, the center of the kernel will have x and y values of 0.
    # We have half_size number of <0 numbers and half_size number of >0 numbers, and the number zero.
    for x in range(-half_size, half_size + 1):
        for y in range(-half_size, half_size + 1):
            value = (1 / (2 * np.pi * sigma ** 2)) * \
                    exp(- (x ** 2 + y ** 2) / (2 * sigma ** 2))
            g_filter[x + half_size, y + half_size] = value

    return g_filter


def sobel_filter(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Performs sobel filtering.

    :param img: 2D numpy array containing a greyscale image
    :return: 2D np.ndarray containing sobel-filtered image
    """
    sobel_filter_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobel_filter_y = np.transpose(sobel_filter_x)

    x_edge_mag = img_filter(img, sobel_filter_x)
    y_edge_mag = img_filter(img, sobel_filter_y)

    edge_mag = np.sqrt(np.square(x_edge_mag) + np.square(y_edge_mag))

    edge_ori = np.zeros(img.shape, dtype=np.float32)
    height, width = img.shape
    for h in range(height):
        for w in range(width):
            # Getting the Edge Orientation takes a couple of lines to improve readability
            edge_ori_in_rads = np.arctan2(y_edge_mag[h][w], x_edge_mag[h][w])
            edge_ori_in_degs = (edge_ori_in_rads * 180 / np.pi) % 180           # mod 180 to only get [0,180] numbers
            rounded_edge_ori = (np.round(edge_ori_in_degs / 45) * 45) % 180     # mod 180 for turning 180 => 0

            edge_ori[h][w] = rounded_edge_ori

    return edge_mag, edge_ori


def non_max_suppression(edge_mag: np.ndarray, edge_ori: np.ndarray) -> np.ndarray:
    """
    Performs Non Max Suppression on a greyscale image using edge magnitude and edge orientation.

    :param edge_mag: 2D numpy array containing edge magnitudes
    :param edge_ori: 2D numpy array containing edge orientations
    :return: 2D numpy array containing image post-nms
    """
    edge_mag_padded = reflect_pad(edge_mag, 1, 1)
    post_nms = np.copy(edge_mag)

    height, width = edge_mag.shape
    for h in range(height):
        for w in range(width):
            # Any point (x,y) in non-padded image correlates to point (x+1, y+1) in padded image
            curr_mag = edge_mag_padded[h + 1][w + 1]

            # Yes, the nested if statements can be removed by using an and operator on the outer if statement, but doing
            # so reduces readability by too much imo.  Sorry but the code is going to be a bit slower. :(
            if edge_ori[h][w] == 0:
                if edge_mag_padded[h + 1][w] >= curr_mag or edge_mag_padded[h + 1][w + 2] >= curr_mag:
                    post_nms[h][w] = 0
            elif edge_ori[h][w] == 45:
                if edge_mag_padded[h][w + 2] >= curr_mag or edge_mag_padded[h + 2][w] >= curr_mag:
                    post_nms[h][w] = 0
            elif edge_ori[h][w] == 90:
                if edge_mag_padded[h][w + 1] >= curr_mag or edge_mag_padded[h + 2][w + 1] >= curr_mag:
                    post_nms[h][w] = 0
            elif edge_ori[h][w] == 135:
                if edge_mag_padded[h + 2][w] >= curr_mag or edge_mag_padded[h][w + 2] >= curr_mag:
                    post_nms[h][w] = 0

    return post_nms


def low_thresholding(img: np.ndarray, threshold: float) -> np.ndarray:
    """
    Applies a low threshold to an image to reduce noise.

    :param img: 2D numpy array containing a greyscale image
    :param threshold: float representing the low threshold - working in [0,1] range
    :return: 2D numpy array containing image after thresholding
    """
    post_threshold = np.copy(img)

    height, width = img.shape
    for h in range(height):
        for w in range(width):
            if post_threshold[h][w] < threshold:
                post_threshold[h][w] = 0

    return post_threshold

def reflect_pad(img: np.ndarray, pad_w: int, pad_h: int) -> np.ndarray:
    """
    Performs Reflection padding on a given image with specified amount of width and height padding

    :param img: 2D numpy array containing a greyscale image
    :param pad_w: amount of width padding to do
    :param pad_h: amount of height padding to do
    :return: 2D numpy array containing padded image
    """
    height, width = img.shape

    # Sets up for recursion reflection
    # Each iteration will only reflect by the size of the input image
    # Is it efficient? Definitely not, but I realized the day before the due date that we can't use np.pad() :`(
    extra_h, extra_w = 0, 0
    if pad_w > width:
        extra_w = pad_w - width
        pad_w = width
    if pad_h > height:
        extra_h = pad_h - height
        pad_h = height

    padded = np.zeros((height + pad_h * 2, width + pad_w * 2))

    # Original Image Copying
    for h in range(height):
        for w in range(width):
            padded[h + pad_h][w + pad_w] = img[h][w]

    # Corner Copying
    for h in chain(range(pad_h), range(height + pad_h, height + 2 * pad_h)):
        if h < pad_h:
            mir_h = 2 * pad_h - h - 1
        else:
            mir_h = 2 * (pad_h + height) - h - 1
        for w in chain(range(pad_w), range(width + pad_w, width + 2 * pad_w)):
            if w < pad_w:
                mir_w = 2 * pad_w - w - 1
            else:
                mir_w = 2 * (pad_w + width) - w - 1

            padded[h][w] = padded[mir_h][mir_w]

    # Edge Padding - without corners
    for i in range(2):
        # Pads along the top edge
        for h in range(pad_h):
            mir_h = 2 * pad_h - h - 1
            for w in range(pad_w, pad_w + width):
                padded[h][w] = padded[mir_h][w]

        # Pads along the left edge
        for w in range(pad_w):
            mir_w = 2 * pad_w - w - 1
            for h in range(pad_h, pad_h + height):
                padded[h][w] = padded[h][mir_w]

        # Rotates the image to pad the bottom and right edge
        padded = np.rot90(padded, 2)

    # Enters recursion if needed
    if extra_w or extra_h:
        padded = reflect_pad(padded, extra_w, extra_h)

    return padded