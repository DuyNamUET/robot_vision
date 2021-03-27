import cv2
import numpy as np
import matplotlib.pyplot as plt

def calc_histogram(gray):
    ''' Calclutate histogram of gray image'''
    hist = np.zeros([256], dtype=np.uint32)
    
    h, w = gray.shape
    for i in range(h):
        for j in range(w):
            hist[gray[i,j]] += 1
    return hist

def show_histogram(hist1, title1="", hist2=None, title2=None):
    ''' Plot histogram'''
    x_axis = np.arange(256)
    fig = plt.figure()
    if hist2 is None:
        plt.bar(x_axis, hist1)
        plt.title(title1)
    else:
        # Plot histogram of hist1
        fig.add_subplot(1, 2, 1)
        plt.bar(x_axis, hist1)
        plt.title(title1)

        # Plot histogram of hist2
        fig.add_subplot(1, 2, 2)
        plt.bar(x_axis, hist2)
        plt.title(title2)
    plt.show()

def make_cumsum(hist):
    ''' Create an array that represents the cumulative sum of the histogram '''
    cumsum = np.zeros(256, dtype=np.uint32)
    cumsum[0] = hist[0]
    for i in range(1, hist.size):
        cumsum[i] = cumsum[i-1] + hist[i]
    return cumsum

def make_mapping(cumsum, img_size):
    ''' Create a mapping s.t. each old colour value is mapped to a new one between 0 and 255 '''
    mapping = np.zeros(256, dtype=np.int32)
    grey_levels = 256
    for i in range(grey_levels):
        mapping[i] = int((grey_levels*cumsum[i])/(img_size[0]*img_size[1]))
    return mapping

def hist_equalization(gray, mapping):
    ''' Apply the mapping to our image '''
    h, w = gray.shape
    out = np.zeros([h,w], dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            out[i,j] = mapping[gray[i,j]]
    return out

if __name__ == "__main__":
    img = cv2.imread("images/brainct-lowcontrast.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hist = calc_histogram(gray)
    cumsum = make_cumsum(hist)
    # print(cumsum)
    
    mapping = make_mapping(cumsum, gray.shape)
    # print(mapping)
    equa_img = hist_equalization(gray, mapping)

    equa_hist = calc_histogram(equa_img)
    
    # show_histogram(hist)
    show_histogram(hist, title1="Origin", hist2=equa_hist, title2="Equalization")

    cv2.imshow("Origin", gray)
    cv2.imshow("Equalization", equa_img)
    cv2.waitKey(0)