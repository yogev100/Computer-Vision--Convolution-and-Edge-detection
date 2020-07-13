import cv2
import matplotlib.pyplot as plt
import numpy as np
from ex2_utils import *

from Ex2 import ex2_utils


def main():
    boxman = cv2.imread('boxman.jpg', cv2.IMREAD_GRAYSCALE)
    beach = cv2.imread('beach.jpg', cv2.IMREAD_GRAYSCALE)
    codeMonkey = cv2.imread('codeMonkey.jpeg', cv2.IMREAD_GRAYSCALE)
    coins = cv2.imread('coins.jpg', cv2.IMREAD_GRAYSCALE)

    print("ID's:", ex2_utils.myID())

    # 1D convolution test
    img_1d = [5, 2, 19, 55, 10, 4, 2, 4, 9, 17, 152, 1]
    kernel_1d = [1, 2, -1]
    conv_1D_test(img_1d, kernel_1d)

    # 2D convolution test
    kernel_2d = np.ones((11,11))
    kernel_2d /= kernel_2d.sum()
    conv_2D_test(beach, kernel_2d)

    # derivative test
    derivative_test(boxman)

    # blurring test
    blur_img1_img2_test(beach, 5)

    # sobel test
    sobel_test(codeMonkey)

    # zero crossing log test
    zero_crossing_simple_Log_test(boxman)

    # canny test
    canny_test(boxman)

    # hough circle test
    hough_transform_circle_test(coins, 40, 108)

def conv_1D_test(img:np.ndarray, kernel1:np.ndarray)->None:
    print("conv_1D_test:")
    ans=ex2_utils.conv1D(img, kernel1)
    ans_cv2 = np.convolve(img, kernel1, 'full')
    print("our implementation:")
    print(ans)
    print("cv2 implementation:")
    print(ans_cv2)
    d = diff(ans,ans_cv2)
    print("difference:", d)

def conv_2D_test(img:np.ndarray, kernel2:np.ndarray)->None:
    print("conv_2D_test:")
    ans = ex2_utils.conv2D(img, kernel2)
    ans_cv2 = cv2.filter2D(img, -1, np.flip(kernel2), borderType=cv2.BORDER_REPLICATE)
    f, ax = plt.subplots(1, 2)
    ax[0].set_title('our conv2D implementaion')
    ax[1].set_title('cv2 conv2D implementaion')
    ax[0].imshow(ans, cmap='gray')
    ax[1].imshow(ans_cv2, cmap='gray')
    plt.show()

def derivative_test(img:np.ndarray)->None:
    print("derivative_test:")
    direct, mag, Ix, Iy = ex2_utils.convDerivative(img)
    f, ax = plt.subplots(1, 3)
    ax[0].set_title('Magnitude')
    ax[1].set_title('Ix Derviative')
    ax[2].set_title('Iy Derviative')
    ax[0].imshow(mag, cmap='gray')
    ax[1].imshow(Ix, cmap='gray')
    ax[2].imshow(Iy, cmap='gray')
    plt.show()

def blur_img1_img2_test(img:np.ndarray,size:np.int)->None:
    print("blur_img1_img2_test:")
    img1 = ex2_utils.blurImage1(img,size)
    img2 = ex2_utils.blurImage2(img,size)
    f, ax = plt.subplots(1, 2)
    ax[0].set_title('blurImage1')
    ax[1].set_title('blurImage2')
    ax[0].imshow(img1, cmap='gray')
    ax[1].imshow(img2, cmap='gray')
    plt.show()

def sobel_test(img:np.ndarray,thresh:float=0.7)->None:
    print("sobel_test:")
    sobel_cv2, sobel_imp = ex2_utils.edgeDetectionSobel(img, thresh)
    f, ax = plt.subplots(1, 2)
    ax[0].set_title('Sobel cv2 implementation')
    ax[1].set_title('Sobel our implementation')
    ax[0].imshow(sobel_cv2, cmap='gray')
    ax[1].imshow(sobel_imp, cmap='gray')
    plt.show()

def zero_crossing_simple_Log_test(img:np.ndarray)->None:
    print("zero_crossing_simple_Log_test:")
    zero_crossing_img_simple = ex2_utils.edgeDetectionZeroCrossingSimple(img)
    zero_crossing_img_log = ex2_utils.edgeDetectionZeroCrossingLOG(img)
    f, ax = plt.subplots(1, 2)
    ax[0].set_title('zero_crossing_img_simple')
    ax[1].set_title('zero_crossing_img_Log')
    ax[0].imshow(zero_crossing_img_simple, cmap='gray')
    ax[1].imshow(zero_crossing_img_log, cmap='gray')
    plt.show()

def canny_test(img:np.ndarray)->None:
    print("canny_test:")
    cv2_canny, our_canny = ex2_utils.edgeDetectionCanny(img)
    f, ax = plt.subplots(1, 2)
    ax[0].set_title('cv2_canny')
    ax[1].set_title('our_canny')
    ax[0].imshow(cv2_canny, cmap='gray')
    ax[1].imshow(our_canny, cmap='gray')
    plt.show()

def hough_transform_circle_test(img:np.ndarray,min_radius:float,max_radius:float)->None:
    print("hough_transform_circle_test:")
    hough_list = ex2_utils.houghCircle(img, min_radius, max_radius)
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    for c in hough_list:
        circle1 = plt.Circle((c[0], c[1]), c[2], color='r', fill=False)
        ax.add_artist(circle1)
    plt.show()

def diff(img1:np.ndarray,img2:np.ndarray)->float:
    dif = np.abs(img1.sum()-img2.sum())
    return dif

if __name__ == '__main__':
    main()