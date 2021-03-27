import cv2
import numpy as np
from matplotlib import pyplot as plt

def omega(p, q):
   ''' The omega term in DFT and IDFT formulas'''
   return np.exp((2.0 * np.pi * 1j * q) / p)

def pad(lst):
   '''padding the list to next nearest power of 2 as FFT implemented is radix 2'''
   k = 0
   while 2**k < len(lst):
      k += 1
   return np.concatenate((lst, ([0] * (2 ** k - len(lst)))))

def fft(x):
   ''' FFT of 1-d signals
   usage : X = fft(x)
   where input x = list containing sequences of a discrete time signals
   and output X = dft of x '''

   n = len(x)
   if n == 1:
      return x
   Feven, Fodd = fft(x[0::2]), fft(x[1::2])
   combined = [0] * n
   for m in range(n//2):
     combined[m] = Feven[m] + omega(n, -m) * Fodd[m]
     combined[m + n//2] = Feven[m] - omega(n, -m) * Fodd[m]
   return combined

def ifft(X):
   ''' IFFT of 1-d signals
   usage x = ifft(X) 
   unpadding must be done implicitly'''

   x = fft([x.conjugate() for x in X])
   return [x.conjugate()/len(X) for x in x]

def pad2(x):
   m, n = np.shape(x)
   M = 2 ** int(np.ceil(np.log2(m)))
   N = 2 ** int(np.ceil(np.log2(n)))
   F = np.zeros((M,N), dtype = x.dtype)
   F[0:m, 0:n] = x
   return F, m, n

def fft2(f):
   '''FFT of 2-d signals/images with padding
   usage X, m, n = fft2(x), where m and n are dimensions of original signal'''

   f, m, n = pad2(f)
   return np.transpose(fft(np.transpose(fft(f)))), m, n

def ifft2(F, m, n):
   ''' IFFT of 2-d signals
   usage x = ifft2(X, m, n) with unpaded, 
   where m and n are odimensions of original signal before padding'''

   f, M, N = fft2(np.conj(F))
   f = np.matrix(np.real(np.conj(f)))/(M*N)
   return f[0:m, 0:n]

def fftshift(F):
   ''' this shifts the centre of FFT of images/2-d signals'''
   M, N = F.shape
   R1, R2 = F[0: M/2, 0: N/2], F[M/2: M, 0: N/2]
   R3, R4 = F[0: M/2, N/2: N], F[M/2: M, N/2: N]
   sF = np.zeros(F.shape,dtype = F.dtype)
   sF[M/2: M, N/2: N], sF[0: M/2, 0: N/2] = R1, R4
   sF[M/2: M, 0: N/2], sF[0: M/2, N/2: N]= R3, R2
   return sF

if __name__ == "__main__":
    img = cv2.imread("images/brain.jpg", 0)

    X,m,n = fft2(img)
    
    f = np.fft.fft2(img)

    plt.subplot(131)
    plt.imshow(img, cmap='gray')
    plt.subplot(132)
    plt.imshow(np.abs(X), cmap='gray')
    plt.subplot(133)
    plt.imshow(np.abs(f), cmap='gray')
    plt.show()

    # kernel = np.array([ [1,1,1],
    #                     [1,1,1],
    #                     [1,1,1]])

    # out = np.zeros([100,100])
    # mask = np.ones([102,102])
    # w = 100; h = 100
    # for i in range(100):
    #     for j in range(100):
    #         out[i,j] = np.sum(kernel*mask[i:i+3, j:j+3])


    # fft_filter = np.fft.fft2(out)
    # fft_shift = np.fft.fftshift(fft_filter)


    # print(np.abs(fft_shift))
    # plt.imshow(np.abs(fft_shift), cmap = 'gray')
    # plt.show()