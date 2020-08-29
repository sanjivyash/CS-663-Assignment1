import os
import math
import numpy as np 
import cv2 as cv
import time
from concurrent.futures import ProcessPoolExecutor as executor
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1 import make_axes_locatable


def extractImage(path):
    img = cv.imread(path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    assert isinstance(img, np.ndarray)
    return img


def presums(arr):
    ans = arr.copy()
    for i in range(1, len(arr)):
        ans[i] = arr[i] + ans[i-1]
    return ans 


def gen_windows(img, N, threshold, m, n):
    for i in range(N//2, (m+N//2)):
        for j in range(N//2, (n+N//2)):
            yield img[(i-N//2):(i+(N//2)+1), (j-N//2):(j+(N//2)+1)], N, threshold, img[i][j], [i,j]


def make_hist(tup):
    window, N, threshold, [r1,g1,b1], [i1,j1] = tup
    print(i1,j1)

    m,n = window.shape[:2]
    
    R = np.zeros(256, dtype='int')
    G, B = R.copy(), R.copy()
    
    for k in range(m):
        for l in range(n):
            r,g,b = window[k][l]
            
            if((r != 300) and (g !=300) and (b !=300)):
                R[r] += 1
                G[g] += 1
                B[b] += 1
    
    R, G, B = (R/(N*N)), (G/(N*N)), (B/(N*N)) 
    sum_r, sum_g, sum_b = np.zeros(3, dtype=np.float64)
    
    for i in range(256):
        if(R[i] > threshold):
            sum_r += R[i]-threshold
            R[i] = threshold
        
        if(G[i] > threshold):
            sum_g += G[i]-threshold
            G[i] = threshold
        
        if(B[i] > threshold):
            sum_b += B[i]-threshold
            B[i] = threshold

    R, G, B = (R + (sum_r/256)), (G + (sum_g/256)), (B + (sum_b/256))
    sR, sG, sB = presums(R), presums(G), presums(B)
    
    return (np.array([sR[r1], sG[g1], sB[b1]]), (i1,j1))


def myCLAHE(path, N, threshold):
    if(isinstance(path, str)):
        img = extractImage(path)
    elif(isinstance(path, np.ndarray)):
        img = path
    else:
        raise TypeError('Enter path or image')
    
    m,n = img.shape[:2]
    
    img_copy = np.ones(((m+2*(N//2)), (n+2*(N//2)), 3), dtype='int')*300
    img_copy[(N//2):(m+N//2), (N//2):(n+N//2)] = img
    
    print(img_copy.shape)
    out = np.zeros(img.shape, dtype=np.float64)

    with executor() as process:
        val = []
        for tup in gen_windows(img_copy, N, threshold, m, n):
            val.append(process.submit(make_hist, tup))
    
    for v in val:
        value, key = v.result()
        [i,j]= key
        out[i-(N//2)][j-(N//2)] = value

    image_titles=["Original Image", "Contrasted Image"]
    
    ax = plt.subplot(1,2,1)
    im = ax.imshow(img)
    ax.set_title(image_titles[0])

    ax = plt.subplot(1,2,2)

    im = ax.imshow(out)
    ax.set_title(image_titles[1])
    divider = make_axes_locatable(ax)
    
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax= cax)


if __name__ == '__main__':
    path = os.path.join(os.path.dirname(__file__), 'canyon', 'canyon.png')
    print(path)

    start = time.time()
    myCLAHE(path, 40, 0.02)
    print(time.time() - start)

    plt.show()