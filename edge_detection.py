import cv2
import numpy as np

def edge_sobel(gray, norm="NORM_L1"):
    '''
    Edge detection using Sobel operator
    gray - input image must be gray scale
    norm - norm type (NORM_L1 | NORM_L2), default: NORM_L1
    '''
    # Expand the raw image
    h, w = gray.shape   # Get image shape
    gray_cp = np.zeros([h+2, w+2], dtype=np.uint8)    # Init the zeros matrix with expand size
    gray_cp[1:1+h, 1:1+w] = gray    # Assign the raw image to new expand image

    # Create sobel operator
    sobel_x = np.array([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=np.intc)     # x direction
    sobel_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=np.intc)     # y direction

    # Init the output image
    out = np.zeros([h,w], dtype=np.uint8)

    # Browse all elements
    for i in range(h):
        for j in range(w):
            local = gray_cp[i:i+3, j:j+3]
            gradient_x = np.sum(local*sobel_x)
            gradient_y = np.sum(local*sobel_y)
            
            if(norm == "NORM_L2"):
                out[i,j] = np.abs(gradient_x) + np.abs(gradient_y)      # norm l2
            else:
                out[i,j] = np.sqrt(gradient_x**2 + gradient_y**2)       # norm l1
    
    return out

def edge_scharr(gray, norm="NORM_L1"):
    '''
    Edge detection using Scharr operator
    gray - input image must be gray scale
    norm - norm type (NORM_L1 | NORM_L2), default: NORM_L1
    '''
    # Expand the raw image
    h, w = gray.shape   # Get image shape
    gray_cp = np.zeros([h+2, w+2], dtype=np.uint8)    # Init the zeros matrix with expand size
    gray_cp[1:1+h, 1:1+w] = gray    # Assign the raw image to new expand image

    # Create sobel operator
    scharr_x = np.array([[-3,0,3],[-10,0,10],[-3,0,3]], dtype=np.intc)     # x direction
    scharr_y = np.array([[3,10,3],[0,0,0],[-3,-10,-3]], dtype=np.intc)     # y direction

    # Init the output image
    out = np.zeros([h,w], dtype=np.uint8)

    # Browse all elements
    for i in range(h):
        for j in range(w):
            local = gray_cp[i:i+3, j:j+3]
            gradient_x = np.sum(local*scharr_x)
            gradient_y = np.sum(local*scharr_y)
            
            if(norm == "NORM_L2"):
                out[i,j] = np.abs(gradient_x) + np.abs(gradient_y)      # norm l2
            else:
                out[i,j] = np.sqrt(gradient_x**2 + gradient_y**2)       # norm l1
    
    return out

def edge_laplacian(gray, ker_type=1):
    '''
    Edge detection using Laplacian operator
    gray - input image must be gray scale
    ker_type - the Laplacian operator type (1 | 2 | 3), default: 1
    '''

    # Create the Laplacian operator types
    l1_ker = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]], dtype=np.intc)         # Type 1
    l2_ker = np.array([[1,-2,1],[-2,4,-2],[1,-2,1]], dtype=np.intc)         # Type 2
    l3_ker = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], dtype=np.intc)     # Type 3

    # Expand the raw image
    h, w = gray.shape   # Get image shape
    gray_cp = np.zeros([h+2, w+2], dtype=np.uint8)    # Init the zeros matrix with expand size
    gray_cp[1:1+h, 1:1+w] = gray    # Assign the raw image to new expand image

    # Init the output image
    out = np.zeros([h,w], dtype=np.uint8)

    # Browse all elements
    for i in range(h):
        for j in range(w):
            local = gray_cp[i:i+3, j:j+3]
            
            if(ker_type == 1):
                out[i,j] = np.sum(local*l1_ker)     
            elif(ker_type == 2):
                out[i,j] = np.sum(local*l2_ker)     
            else:
                out[i,j] = np.sum(local*l3_ker)     
                
    return np.abs(out)


if __name__ == "__main__":
    # Read image
    img = cv2.imread("images/lumbar_spine.jpg")

    # Convert image to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect edge with Sobel
    # sobel = edge_sobel(gray, norm="NORM_L2")

    # Detect edge with Laplacian
    # laplacian = edge_laplacian(gray, ker_type=1)

    # Show
    cv2.imshow("Original", gray)
    # cv2.imshow("Sobel Edge Detection", sobel)
    # cv2.imshow("Laplacian Edge Detection", laplacian)
    cv2.waitKey(0)