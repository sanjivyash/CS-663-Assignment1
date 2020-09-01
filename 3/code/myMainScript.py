import os
import math
import numpy as np 
import cv2 as cv
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1 import make_axes_locatable


def extractImage(path):
  img = cv.imread(path)
  img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  assert isinstance(img, np.ndarray)
  return img


def presums(arr):
  ans = arr.copy()
  for i in range(1, len(arr)):
    ans[i] = arr[i] + ans[i-1]
  return ans 


def myHE(path):
  if isinstance(path, str):
    img = extractImage(path)
  elif isinstance(path, np.ndarray):
    img = path
  else:
    raise TypeError('Provide image or path')

  m, n = img.shape[:2]
  R = [0 for i in range(256)]

  for i in range(m):
    for j in range(n):
      r = img[i][j]
      R[r] += 1

  sR= np.array(presums(R))
  fR=sR/sR[-1]

  out = np.zeros(img.shape, dtype=np.float64)

  for i in range(m):
    for j in range(n):
      r = img[i][j]
      out[i][j] = fR[r]

  return out
  

def myNewHE(path):
  if(isinstance(path, str)):
    img=extractImage(path)
  elif(isinstance(path, np.ndarray)):
    img=path
  else:
    TypeError('Path or Image object not given')
  
  m,n = img.shape[:2]
  R = np.zeros(256, dtype=np.int)

  for i in range(m):
    for j in range(n):
      r = img[i][j]
      R[r] += 1

  sR= presums(R)
  fR= [i/(m*n) for i in sR]
  median = 0
  
  for i in range(len(fR)):
    if(fR[i]>=0.5):
      median=i
      break
  
  print(median)

  R1 = R[:median+1]
  sR1 = np.array(presums(R1))
  fR1 = (sR1/sR1[-1])*(median/255)
  
  R2 = R[median+1:]
  sR2 = np.array(presums(R2))
  fR2 = ((median)/255) + (sR2/sR2[-1])*((255-median)/255)
  
  print("FR1: ", fR1)
  print('FR2 :', fR2)

  out = np.zeros(img.shape, dtype=np.float64)
  outHE = myHE(path)

  for i in range(m):
    for j in range(n):
      r=img[i][j]
      
      if(r<=median):
        out[i][j] = fR1[r]
      else:
        out[i][j] = fR2[r-median-1]

  image_titles=["Original Image", "HE image", "Median HE Image"]

  ax = plt.subplot(1,3,1)
  im = ax.imshow(img, cmap='gray')
  ax.set_title(image_titles[0])

  ax = plt.subplot(1,3,2)

  im = ax.imshow(outHE, cmap="gray")
  ax.set_title(image_titles[1])

  ax = plt.subplot(1,3,3)

  im = ax.imshow(out, cmap="gray")
  ax.set_title(image_titles[2])

  divider = make_axes_locatable(ax)
  cax = divider.append_axes('right', size='5%', pad=0.05)

  plt.colorbar(im, cax= cax)
  plt.tight_layout()

  plt.show()


if __name__ == '__main__':
  path = os.path.join(os.path.dirname(_file_), '..', 'data', 'camera.png')
  myNewHE(path)