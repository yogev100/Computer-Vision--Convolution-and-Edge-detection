import matplotlib.pyplot as plt
import numpy as np
import cv2
import Ex2

def main():
    #1D convolution test
    img_1d=[5,2,19,55,10,4,2,4,9,17,152,1]
    plt.gray()
    kernel_1d=[1,2,-1]
    conv_1D_test(img_1d,kernel_1d)

    #2D convolution test
    # img=cv2.imread('flowers.jpg',cv2.IMREAD_GRAYSCALE)
    # kernel_2d=np.ones((11,11))
    # kernel_2d/=kernel_2d.sum()
    # conv_2D_test(img,kernel_2d)

    img=cv2.imread('two.jpg',cv2.IMREAD_GRAYSCALE)
    plt.gray()

    #derivative test
    derivative_test(img)

    # sobel test
    sobel_test(img)

    # zero crossing log test
    zero_crossing_log_test(img)








def conv_1D_test(img:np.ndarray,kernel1:np.ndarray)->None:
    print("conv_1D_test:")
    ans=Ex2.conv1D(img,kernel1)
    ans_cv2=np.convolve(img,kernel1,'full')
    print("our implementation:")
    print(ans)
    print("cv2 implementation:")
    print(ans_cv2)
    d= diff(ans,ans_cv2)
    print("difference:",d)

def conv_2D_test(img:np.ndarray,kernel2:np.ndarray)->None:
    print("conv_2D_test:")
    ans=Ex2.conv2D(img,kernel2)
    ans_cv2=cv2.filter2D(img,-1,np.flip(kernel2),borderType=cv2.BORDER_REPLICATE)
    f, ax = plt.subplots(1,2)
    ax[0].imshow(ans)
    ax[1].imshow(ans_cv2)
    plt.show()

def derivative_test(img:np.ndarray)->None:
    print("derivative_test:")
    direct, mag, Ix, Iy=Ex2.convDerivative(img)
    f, ax = plt.subplots(1, 3)
    ax[0].imshow(mag)
    ax[1].imshow(Ix)
    ax[2].imshow(Iy)
    plt.show()

def sobel_test(img:np.ndarray,thresh:float=0.7)->None:
    print("sobel_test:")
    sobel_cv2,sobel_imp=Ex2.edgeDetectionSobel(img,thresh)
    f, ax = plt.subplots(1, 2)
    ax[0].imshow(sobel_cv2)
    ax[1].imshow(sobel_imp)
    plt.show()

def zero_crossing_log_test(img:np.ndarray)->None:
    print("zero_crossing_log_test:")
    crossing_img=Ex2.edgeDetectionZeroCrossingLOG(img)
    plt.imshow(crossing_img)
    plt.show()

def diff(img1:np.ndarray,img2:np.ndarray)->float:
    diff=np.abs(img1.sum()-img2.sum())
    return diff

if __name__ == '__main__':
    main()