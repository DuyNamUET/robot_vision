import cv2
import numpy as np

def inverse_image(gray):
    '''
    Convert the gray image to it's inverse image
    gray - input image must be gray scale
    '''
    h, w = gray.shape
    # print("{} {}".format(w,h))
    mask = 255*np.ones([h,w], dtype=np.uint8)
    i_img = mask - gray
    return i_img

if __name__ == "__main__":
    # Read image
    img = cv2.imread("images/brain.jpg")

    # Convert image to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Inverse the image
    i_img = inverse_image(gray)

    # Show
    cv2.imshow("Original", gray)
    cv2.imshow("Inverse Image", i_img)
    cv2.waitKey(0)