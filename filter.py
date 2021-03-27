import cv2
import numpy as np

def median_filter(gray, kernel=3, copy_border=False):
    '''
    Filter the image with salt and pepper noise
    gray - input image must be gray scale
    kernel - the kernel size (default: 3)
    copy_border - copy border image to expand image before filter (default: False)
    '''
    # Expand the raw image
    h, w = gray.shape   # Get image shape
    gray_cp = np.zeros([h+kernel-1, w+kernel-1], dtype=np.uint8)    # Init the zeros matrix with expand size
    gray_cp[kernel//2:kernel//2+h, kernel//2:kernel//2+w] = gray    # Assign the raw image to new expand image
    
    if(copy_border):
        gray_cp[0:kernel//2, kernel//2:kernel//2+w] = gray[:kernel//2,:]    # Copy to top
        gray_cp[kernel//2+h:, kernel//2:kernel//2+w] = gray[h-kernel//2,:]  # Copy to bottom
        gray_cp[kernel//2:kernel//2+h, 0:kernel//2] = gray[:,:kernel//2]    # Copy to left
        gray_cp[kernel//2:kernel//2+h, kernel//2+w:] = gray[:,w-kernel//2:] # Copy to right


    median_img = np.zeros([h, w], dtype=np.uint8) # Init the output image

    # Browse all elements
    for i in range(h):
        for j in range (w):
            local = gray_cp[i:i+kernel, j:j+kernel] # Init the local matrix with same size with kernel matrix
            median_img[i,j] =  np.sort(local, axis=None)[kernel**2//2]  # Sort the local matrix and get the median value
    return median_img

def mean_filter(gray, kernel=3, copy_border=False):
    '''
    Filter image with mean
    gray - input image must be gray scale
    kernel - the kernel size (default: 3)
    copy_border - copy border image to expand image before filter (default: False)
    '''
    # Create the kernel matrix
    ker_mat = np.ones((kernel,kernel))/(kernel**2)

    # Expand the raw image
    h, w = gray.shape   # Get image shape
    gray_cp = np.zeros([h+kernel-1, w+kernel-1], dtype=np.uint8)    # Init the zeros matrix with expand size
    gray_cp[kernel//2:kernel//2+h, kernel//2:kernel//2+w] = gray    # Assign the raw image to new expand image
    
    if(copy_border):
        gray_cp[0:kernel//2, kernel//2:kernel//2+w] = gray[:kernel//2,:]    # Copy to top
        gray_cp[kernel//2+h:, kernel//2:kernel//2+w] = gray[h-kernel//2,:]  # Copy to bottom
        gray_cp[kernel//2:kernel//2+h, 0:kernel//2] = gray[:,:kernel//2]    # Copy to left
        gray_cp[kernel//2:kernel//2+h, kernel//2+w:] = gray[:,w-kernel//2:] # Copy to right

    mean_img = np.zeros([h, w], dtype=np.uint8) # Init the output image

    # Browse all elements
    for i in range(h):
        for j in range (w):
            local = gray_cp[i:i+kernel, j:j+kernel] # Init the local matrix with same size with kernel matrix
            mean_img[i,j] = np.sum(local*ker_mat)   # Sum of the scalar product
    return mean_img

def gaussian_filter(gray, kernel=3, copy_border=False):
    '''
    Filter image with Gaussian 
    gray - input image must be gray scale
    kernel - the kernel size (default: 3)
    copy_border - copy border image to expand image before filter (default: False)
    '''
    # Create the Gauss kernel
    size = kernel//3
    x, y = np.mgrid[-size:size+1, -size:size+1]
    ker_mat = np.exp(-(x**2/size + y**2/size))
    ker_mat /= ker_mat.sum()

    # Expand the raw image
    h, w = gray.shape   # Get image shape
    gray_cp = np.zeros([h+kernel-1, w+kernel-1], dtype=np.uint8)    # Init the zeros matrix with expand size
    gray_cp[kernel//2:kernel//2+h, kernel//2:kernel//2+w] = gray    # Assign the raw image to new expand image
    
    if(copy_border):
        gray_cp[0:kernel//2, kernel//2:kernel//2+w] = gray[:kernel//2,:]    # Copy to top
        gray_cp[kernel//2+h:, kernel//2:kernel//2+w] = gray[h-kernel//2,:]  # Copy to bottom
        gray_cp[kernel//2:kernel//2+h, 0:kernel//2] = gray[:,:kernel//2]    # Copy to left
        gray_cp[kernel//2:kernel//2+h, kernel//2+w:] = gray[:,w-kernel//2:] # Copy to right

    gauss_img = np.zeros([h, w], dtype=np.uint8) # Init the output image

    # Browse all elements
    for i in range(h):
        for j in range (w):
            local = gray_cp[i:i+kernel, j:j+kernel] # Init the local matrix with same size with kernel matrix
            gauss_img[i,j] = np.sum(local*ker_mat)   # Sum of the scalar product
    return gauss_img

if __name__ == "__main__":
    # Read image
    img = cv2.imread("images/brain.jpg")

    # Convert image to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # # Median filter
    # median_img = median_filter(gray, kernel=5)

    # # Mean filter
    # mean_img = mean_filter(gray, copy_border=True)

    # Gaussian filter
    gauss_img = gaussian_filter(gray)

    # Show
    cv2.imshow("Original", gray)
    # cv2.imshow("Median Image", median_img)
    # cv2.imshow("Mean Image", mean_img)
    cv2.imshow("Gauss Image", gauss_img)
    cv2.waitKey(0)