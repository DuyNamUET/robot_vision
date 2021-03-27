import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_fft_image(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    return fshift

def simple_low_pass_filter(img, percent=0.5):
    fshift = get_fft_image(img)
    
    rows, cols = img.shape
    midr = rows//2; midc = cols//2

    mask = np.zeros([rows, cols], dtype=np.uint8)
    mask[int(midr-percent*rows//2):int(midr+percent*rows//2), int(midc-percent*cols//2):int(midc+percent*cols//2)] = 1
    
    fshift = fshift*mask
    f_ishift = np.fft.ifftshift(fshift)
    out = np.fft.ifft2(f_ishift)
    
    return np.abs(out)

def simple_high_pass_filter(img, percent=0.5):
    fshift = get_fft_image(img)
    
    rows, cols = img.shape
    midr = rows//2; midc = cols//2

    fshift[int(midr-percent*rows//2):int(midr+percent*rows//2), int(midc-percent*cols//2):int(midc+percent*cols//2)] = 0
    
    f_ishift = np.fft.ifftshift(fshift)
    out = np.fft.ifft2(f_ishift)
    
    return np.abs(out)

def low_pass_filter(img, distance=50, filter_type="ideal", order=2):
    rows, cols = img.shape
    midr = rows//2; midc = cols//2

    mask = np.zeros([rows, cols])
    for i in range(rows):
        for j in range(cols):
            mask[i,j] = np.sqrt((i-midr)**2 + (j-midc)**2)
    
    if(filter_type == "gauss"):
        mask = np.exp(-mask**2/(2*distance**2))
    elif(filter_type == "btw"):
        mask = 1/(1 + (mask/distance)**(2*order))
    else:
        for i in range(rows):
            for j in range(cols):
                if (mask[i,j] > distance):
                    mask[i,j] = 0
                else:
                    mask[i,j] = 1

    fshift = get_fft_image(img)
    fshift = fshift*mask
    f_ishift = np.fft.ifftshift(fshift)
    out = np.fft.ifft2(f_ishift)
    return np.abs(out)

def high_pass_filter(img, distance=50, filter_type="ideal", order=2):
    rows, cols = img.shape
    midr = rows//2; midc = cols//2

    mask = np.zeros([rows, cols])
    for i in range(rows):
        for j in range(cols):
            mask[i,j] = np.sqrt((i-midr)**2 + (j-midc)**2)
    
    if(filter_type == "gauss"):
        mask = 1 - np.exp(-mask**2/(2*distance**2))
    elif(filter_type == "btw"):
        mask = 1 - 1/(1 + (mask/distance)**(2*order))
    else:
        for i in range(rows):
            for j in range(cols):
                if (mask[i,j] > distance):
                    mask[i,j] = 1
                else:
                    mask[i,j] = 0

    fshift = get_fft_image(img)
    fshift = fshift*mask
    f_ishift = np.fft.ifftshift(fshift)
    out = np.fft.ifft2(f_ishift)
    return np.abs(out)

if __name__ == "__main__":
    img = cv2.imread("images/hiu.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = low_pass_filter(gray, distance=30, filter_type="ideal")
    edge = high_pass_filter(gray, distance=20, filter_type="ideal")

    # blur = simple_low_pass_filter(gray, percent=0.1)
    # edge = simple_high_pass_filter(gray, percent=0.1)

    plt.subplot(131), plt.imshow(gray, cmap = 'gray')
    plt.title("Origin")
    plt.subplot(132); plt.imshow(blur, cmap="gray")
    plt.title("Low pass filter")
    plt.subplot(133); plt.imshow(edge, cmap="gray")
    plt.title("High pass filter")
    plt.show()
