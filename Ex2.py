import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy import signal, ndimage


def main():
    img = cv2.imread('boxman.jpg',cv2.IMREAD_GRAYSCALE)
    plt.gray()
    a,b,c,d=convDerivative(img)
    f, ax = plt.subplots(1,3)
    ax[0].imshow(b)
    ax[1].imshow(c)
    ax[2].imshow(d)
    plt.show()



def conv1D(inSignal:np.ndarray,kernel1:np.ndarray)->np.ndarray:
    res = np.zeros(len(inSignal)+len(kernel1)-1)
    flipedker = np.flip(kernel1)
    sum=0
    for i in range(0,len(inSignal)+len(kernel1)-1):
        if i < len(kernel1)-1:
            ind = i
            ind2 = len(kernel1)-1
            for j in range(i+1, 0,-1):
                sum += flipedker[ind2]*inSignal[ind]
                ind -= 1
                ind2 -= 1
        elif i > len(inSignal)-1:
            ind = len(inSignal)-(len(res)-i)
            ind2 = 0
            for j in range(0, len(res)-i):
                sum += flipedker[ind2]*inSignal[ind]
                ind += 1
                ind2 += 1
        else :
            ind=i
            for j in range(len(kernel1)-1,-1,-1):
                sum += flipedker[j]*inSignal[ind]
                ind -= 1
        res[i] = sum
        sum = 0
    return res

def conv2D(inImage:np.ndarray,kernel2:np.ndarray)->np.ndarray:
    res=np.copy(inImage)
    row=len(kernel2)
    col=len(kernel2[0])

    res1= np.zeros((len(inImage)+2*int(row/2),len(inImage[0])+2*int(col/2)))

    res1[int(row/2):-int(row/2),int(col/2):-int(col/2)]=res

    res1[:int(row/2),:]=res1[int(row/2),:]
    res1[:,:int(col/2)]=np.reshape((res1[:,int(col/2)]),(len(res1),1))
    res1[len(res1)-int(row/2):,:]=res1[len(res1)-int(row/2)-1,:]
    res1[:,len(res1[0])-int(col/2):]=np.reshape((res1[:,len(res1[0])-int(col/2)-1]),(len(res1),1))
    rescpy=res1.copy()

    flipedker = np.flip(kernel2)
    for i in range(0,len(res1)-row+1):
        for j in range(0,len(res1[0])-col+1):
            res1[i+int(row/2)][j+int(col/2)] = np.sum(rescpy[i:i + row, j:j + col] * flipedker)



    return res1[int(row/2):-int(row/2),int(col/2):-int(col/2)]

def convDerivative(inImage:np.ndarray) -> (np.ndarray,np.ndarray,np.ndarray,np.ndarray):
    gaus=cv2.GaussianBlur(inImage,(5,5),1)
    x=np.array([1,0,-1])[np.newaxis]
    Ix = cv2.filter2D(gaus, cv2.CV_64F, np.flip(x), borderType=cv2.BORDER_REPLICATE)
    Iy = cv2.filter2D(gaus, cv2.CV_64F, np.flip(np.transpose(x)), borderType=cv2.BORDER_REPLICATE)
    mag = np.sqrt(np.power(Ix, 2) + np.power(Iy, 2))
    direct=np.arctan2(Iy,Ix)* 180 / np.pi
    return direct,mag,Ix,Iy

def blurImage1(in_image:np.ndarray,kernel_size:int)->np.ndarray:
    kernel=get_gaussian_kernel(kernel_size)
    blurr_img=conv2D(in_image,kernel)
    return blurr_img

def blurImage2(in_image:np.ndarray,size:int)->np.ndarray:
    gaus = cv2.GaussianBlur(in_image, (size, size), 1)
    return gaus

def get_gaussian_kernel(size:int,sigma:float=1)->np.ndarray:
    center = (int)(size / 2)
    kernel = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            diff = np.sqrt((i - center) ** 2 + (j - center) ** 2)
            kernel[i, j] = np.exp(-(diff ** 2) / (2 * sigma ** 2))
    return kernel / np.sum(kernel)

def edgeDetectionSobel(img: np.ndarray, thresh: float = 0.7)-> (np.ndarray, np.ndarray):
    #sobel manual#
    sobelx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float)
    sobely = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float)

    Ix = cv2.filter2D(img, cv2.CV_64F, np.flip(sobelx), borderType=cv2.BORDER_REPLICATE)
    Iy = cv2.filter2D(img, cv2.CV_64F, np.flip(sobely), borderType=cv2.BORDER_REPLICATE)
    iyz=np.zeros_like(Iy)
    ixz = np.zeros_like(Ix)
    iyz[Iy>thresh]=255
    ixz[Ix>thresh]=255
    mag=np.sqrt(np.power(Ix,2)+np.power(Iy,2))

    #sobel with cv2#
    gx_cv2 = cv2.Sobel(img,  cv2.CV_64F, 0, 1, thresh)
    gy_cv2 = cv2.Sobel(img, cv2.CV_64F, 1, 0, thresh)
    mag_cv2=cv2.magnitude(gx_cv2,gy_cv2)
    return mag_cv2,mag

def edgeDetectionZeroCrossingSimple(img:np.ndarray)->(np.ndarray):
    # Apply Laplacian operator in some higher datatype
    laplacian = cv2.Laplacian(img, cv2.CV_64F)

    # Build the new image using zero_crossing function
    new_img=zero_crossing(laplacian)
    return new_img

def edgeDetectionZeroCrossingLOG(img:np.ndarray)->(np.ndarray):
    # Apply Gaussian Blur
    blur = cv2.GaussianBlur(img, (3, 3), 0)

    # Apply Laplacian operator in some higher datatype
    laplacian = cv2.Laplacian(blur, cv2.CV_64F)

    # Build the new image using zero_crossing function
    new_img=zero_crossing(laplacian)
    return new_img

def zero_crossing(img:np.ndarray)->np.ndarray:
    imgc=img/img.max()
    new_img=np.zeros_like(img)
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            negative_count = 0
            positive_count = 0
            neighbour = [imgc[i + 1, j - 1], imgc[i + 1, j], imgc[i + 1, j + 1], imgc[i, j - 1], imgc[i, j + 1],
                         imgc[i - 1, j - 1], imgc[i - 1, j], imgc[i - 1, j + 1]]
            d = max(neighbour)
            e = min(neighbour)
            for h in neighbour:
                if h > 0:
                    positive_count += 1
                elif h < 0:
                    negative_count += 1

            # If both negative and positive values exist in
            # the pixel neighborhood, then that pixel is a
            # potential zero crossing

            z_c = ((negative_count > 0) and (positive_count > 0))

            # Change the pixel value with the maximum neighborhood
            # difference with the pixel

            if z_c:
                if imgc[i, j] > 0:
                    new_img[i, j] = imgc[i, j] + np.abs(e)
                elif imgc[i, j] < 0:
                    new_img[i, j] = np.abs(imgc[i, j]) + d


    # Normalize and change datatype to 'uint8' (optional)
    z_c_norm = new_img / new_img.max() * 255
    z_c_image = np.uint8(z_c_norm)

    return z_c_image

def sobel_filters(img,thresh:float=0.7):
    gx_cv2 = cv2.Sobel(img, cv2.CV_64F, 0, 1, thresh)
    gy_cv2 = cv2.Sobel(img, cv2.CV_64F, 1, 0, thresh)
    mag_cv2 = cv2.magnitude(gx_cv2, gy_cv2)
    direct = np.arctan2(gx_cv2, gy_cv2) * 180 / np.pi
    return mag_cv2, direct

def edgeDetectionCanny(img: np.ndarray, thrs_1: float=100, thrs_2: float=50)-> (np.ndarray, np.ndarray):
    # my implementation
    mag,directions=sobel_filters(img)
    plt.imshow(mag)
    plt.show()
    after_supp_img=non_max_suppression(mag,directions)
    plt.imshow(after_supp_img)
    plt.show()
    after_thresh_img,weak,strong=threshold(after_supp_img,thrs_2,thrs_1)
    plt.imshow(after_thresh_img)
    plt.show()
    after_hys=hysteresis(after_thresh_img,weak,strong)
    #cv2
    cv=cv2.Canny(img,50,100)
    return cv,after_hys

def non_max_suppression(img:np.ndarray, D):
    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.int32)
    angle = D
    angle[angle < 0] += 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            try:
                q = 255
                r = 255

                # angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = img[i, j + 1]
                    r = img[i, j - 1]

                # angle 45
                elif (22.5 <= angle[i, j] < 67.5):
                    q = img[i - 1, j - 1]
                    r = img[i + 1, j + 1]
                # angle 90
                elif (67.5 <= angle[i, j] < 112.5):
                    q = img[i + 1, j]
                    r = img[i - 1, j]

                # angle 135
                elif (112.5 <= angle[i, j] < 157.5):
                    q = img[i + 1, j - 1]
                    r = img[i - 1, j + 1]

                if (img[i, j] >= q) and (img[i, j] >= r):
                    Z[i, j] = img[i, j]
                else:
                    Z[i, j] = 0

            except IndexError as e:
                pass
    return Z

def threshold(img:np.ndarray, lowThresholdRatio=0.05, highThresholdRatio=0.09):
    highThreshold = highThresholdRatio
    lowThreshold = lowThresholdRatio

    M, N = img.shape
    res = np.zeros((M, N), dtype=np.int32)

    weak = np.int32(25)
    strong = np.int32(255)

    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)

    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

    res[zeros_i,zeros_j]=0
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return (res, weak, strong)

def hysteresis(img,weak, strong=255):
    M, N = img.shape
    eps = 3
    for i in range(eps, M-eps):
        for j in range(eps, N-eps):
            if (img[i,j] == weak):
                try:
                    neighbors=img[i-eps:i+eps,j-eps:j+eps]
                    row,col=np.where(neighbors==strong)
                    if len(row)>0:
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img

def houghCircle(img:np.ndarray,min_radius:float,max_radius:float)->list:
    circles_list=[]
    _,direct=sobel_filters(img)
    direct=np.radians(direct)
    any_circle=np.zeros((len(img),len(img[0]),max_radius+1))
    canny_img=cv2.Canny(img,50,100)
    for x in range(0,len(canny_img)):
        print(x)
        for y in range(0,len(canny_img[0])):
            if (canny_img[x][y] > 0):
                for r in range(min_radius,max_radius+1):
                    cy1 = int(y+ r * np.sin(direct[x, y]-np.pi/2))
                    cx1 = int(x - r * np.cos(direct[x, y]-np.pi/2))
                    cy2 = int(y - r * np.sin(direct[x, y]-np.pi/2))
                    cx2 = int(x + r * np.cos(direct[x, y]-np.pi/2))
                    if 0 < cx1 < len(any_circle) and 0 < cy1 < len(any_circle[0]):
                        any_circle[cx1,cy1,r]+=1
                    if 0 < cx2 < len(any_circle) and 0 < cy2 < len(any_circle[0]):
                        any_circle[cx2,cy2,r]+=1

    print(any_circle.max())
    thresh = 0.50*any_circle.max()
    b_center,a_center,radius=np.where(any_circle>=thresh)
    print(b_center)
    print(a_center)
    print(radius)
    for i in range(0,len(a_center)):
        circles_list.append((a_center[i],b_center[i],radius[i]))

    eps=12

    ans=[]
    while len(circles_list)>0:
        temp=circles_list[0]
        index_i=np.where(temp-eps <= circles_list <= temp+eps)

        if len(index_i)>0:
            ans.append(circles_list[index_i[0]])
            del circles_list[index_i]

        # temp_list=[]
        # for i in range(0,len(circles_list)):
        #     if temp[0]-eps <= circles_list[i][0] <= temp[0]+eps and temp[1]-eps <= circles_list[i][1] <= temp[1]+eps and temp[2]-eps <= circles_list[i][2] <= temp[2]+eps:
        #         temp_list.append(circles_list[i])
        #         del circles_list[i]

        # if len(temp_list)>0:
        #     ans.append(temp_list[0])
        #
        # del temp_list[:]




        # if i > len(circles_list)-1:
        #     break
        # temp=circles_list[i]
        # print(i)
        # l=len(circles_list)
        # j=0
        # for k in range(i+1,len(circles_list)-1):
        #     if l!= len(circles_list):
        #         k-=j
        #         l=len(circles_list)
        #     if k==len(circles_list):
        #         break
        #     if temp[0]-eps <= circles_list[k][0] <= temp[0]+eps and temp[1]-eps <= circles_list[k][1] <= temp[1]+eps and temp[2]-eps <= circles_list[k][2] <= temp[2]+eps:
        #         del circles_list[k]
        #         j += 1
        #     else:
        #         j=0


    return ans

def diff(img1:np.ndarray,img2:np.ndarray)->float:
    diff=np.abs(img1.sum()-img2.sum())
    return diff

if __name__ == '__main__':
    main()
