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


def get_cdf(image, threshold, N):
  m, n = image.shape[:2]
  
  R = np.zeros(256, dtype=np.float64)
  G, B = R.copy(), R.copy()

  for i in range(m):
    for j in range(n):
      r, g, b = image[i][j]
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

  return sR, sG, sB


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


def myBilinearCLAHE(path, N, threshold):
  if(isinstance(path, str)):
    img = extractImage(path)
  elif(isinstance(path, np.ndarray)):
    img = path
  else:
    raise TypeError('Enter path or image')

  m,n = img.shape[:2]
  out = np.zeros(img.shape, dtype=np.float64)

  cdf_m = (m//N) + (1 if (m%N!=0) else 0)
  cdf_n = (n//N) + (1 if (n%N!=0) else 0)
  cdfs = np.ndarray((cdf_m, cdf_n, 3, 256), dtype=np.float64)
 
  for i in range(cdf_m):
        for j in range(cdf_n):
            cdfs[i][j] = get_cdf(img[i:(i+N), j:(j+N)], threshold, N)
  
  for i in range(m):
    for j in range(n):
      
      if(i<(N//2) or (j<(N//2))):
        fR, fG, fB = cdfs[i//N][j//N]
        r,g,b = img[i][j]
        out[i][j] = np.array([fR[r], fG[g], fB[b]])
      
      elif(i>(N-(N//2))):
        fR, fG, fB = cdfs[cdf_m-1][j//N]
        r,g,b = img[i][j]
        out[i][j] = np.array([fR[r], fG[g], fB[b]])
      
      elif(j>(N-(N//2))):
        fR, fG, fB = cdfs[i//N][cdf_n-1]
        r,g,b = img[i][j]
        out[i][j] = np.array([fR[r], fG[g], fB[b]])
      
      else:
        fR1, fG1, fB1 = cdfs[i//N][j//N]
        fR2, fG2, fB2 = cdfs[i//N][j//N + 1]
        fR3, fG3, fB3 = cdfs[i//N + 1][j//N]
        fR4, fG4, fB4 = cdfs[i//N + 1][j//N + 1]

        r,g,b = img[i][j]
        
        r1,b1,g1 = fR1[r], fG1[g], fB1[b] 
        r2,b2,g2 = fR2[r], fG2[g], fB2[b] 
        r3,b3,g3 = fR3[r], fG3[g], fB3[b] 
        r4,b4,g4 = fR4[r], fG4[g], fB4[b] 
        
        term_r = (((((i//N) + N - i)*((j//N) + N - j))/(N*N))*r4) + \
            (((((i//N) + N - i)*((j//N) - j))/(N*N))*r3) + \
            (((((i//N) - i)*((j//N) + N - j))/(N*N))*r2) + \
            (((((i//N) - i)*((j//N) - j))/(N*N))*r1)
        
        term_g = (((((i//N) + N - i)*((j//N) + N - j))/(N*N))*g4) + \
            (((((i//N) + N - i)*((j//N) - j))/(N*N))*g3) + \
            (((((i//N) - i)*((j//N) + N - j))/(N*N))*g2) + \
            (((((i//N) - i)*((j//N) - j))/(N*N))*g1)
        
        term_b = (((((i//N) + N - i)*((j//N) + N - j))/(N*N))*b4) + \
            (((((i//N) + N - i)*((j//N) - j))/(N*N))*b3) + \
            (((((i//N) - i)*((j//N) + N - j))/(N*N))*b2) + \
            (((((i//N) - i)*((j//N) - j))/(N*N))*b1)            
        
        out[i][j] = np.array([term_r, term_g, term_b])
        
  image_titles=["Original Image", "Contrasted Image"]
  
  ax = plt.subplot(1,2,1)
  im = ax.imshow(img, cmap='gray')
  ax.set_title(image_titles[0])

  ax = plt.subplot(1,2,2)
  
  im = ax.imshow(out, cmap='gray')
  ax.set_title(image_titles[1])
  
  divider = make_axes_locatable(ax)
  cax = divider.append_axes('right', size='5%', pad=0.05)
  
  plt.colorbar(im, cax= cax)
  plt.tight_layout()


def myExactCLAHE(path, N, threshold):
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

  return out


if __name__ == '__main__':
  path = os.path.join(os.path.dirname(__file__), '..', 'data', 'barbara.png')
  print(path)

  start = time.time()

  # myBilinearCLAHE(path, 20, 0.02) # window size and threshold
  myExactCLAHE(path, 64, 0.02) # window size and threshold
  
  print(time.time() - start)
  plt.show()