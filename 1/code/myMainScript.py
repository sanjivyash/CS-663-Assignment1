import os
import math
import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt 


def myShrinkImageByFactorD(path, scale):
	img = cv.imread(path)
	img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
	assert isinstance(img, np.ndarray)

	n = img.shape[0]
	m = int(n/scale)

	out = np.zeros((m,m,3), dtype=np.uint8)

	for i in range(m):
		for j in range(m):
			out[i][j] = img[i * scale][j * scale]

	plt.figure()
	plt.imshow(img)

	plt.figure()
	plt.imshow(out)

	plt.show()


def myBilinearInterpolation(path, hor, ver):
	img = cv.imread(path)
	img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
	assert isinstance(img, np.ndarray)

	m, n = img.shape[:2]
	M, N = hor*m - (hor-1), ver*n - (ver-1)

	out = np.zeros((M,N,3), dtype=np.float64)

	for i in range(M):
		for j in range(N):
			r, s = 1-(i%hor)/hor, 1-(j%ver)/ver 
			row, col = int(i/hor), int(j/ver)

			out[i][j] = r*s*img[row][col]
			
			if row != m-1:
				out[i][j] += r*(1-s)*img[row+1][col]
			if col != n-1:
				out[i][j] += (1-r)*s*img[row][col+1]

			if row != m-1 and col != n-1:
				out[i][j] += (1-r)*(1-s)*img[row+1][col+1]

			out[i][j] /= 255


	plt.figure()
	plt.imshow(img)

	plt.figure()
	plt.imshow(out)

	plt.show()


def myNearestNeighborInterpolation(path, hor, ver):
	img = cv.imread(path)
	img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
	assert isinstance(img, np.ndarray)

	m, n = img.shape[:2]
	M, N = hor*m - (hor-1), ver*n - (ver-1)

	out = np.zeros((M,N,3), dtype=np.uint8)

	for i in range(M):
		for j in range(N):
			r, s = 1-(i%hor)/hor, 1-(j%ver)/ver 
			row = int(i/hor) if r >= 0.5 else int(i/hor)+1
			col = int(j/ver) if s >= 0.5 else int(j/ver)+1
			out[i][j] = img[row][col]

	plt.figure()
	plt.imshow(img)

	plt.figure()
	plt.imshow(out)

	plt.show()


############################################################
# INCOMPLETE
def myBicubicInterpolation(path, ver, hor):
	img = cv.imread(path)
	img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
	assert isinstance(img, np.ndarray)

	m, n = img.shape[:2]
	M, N = hor*m - (hor-1), ver*n - (ver-1)

	out = np.zeros((M,N,3), dtype=np.uint8)

	for i in range(M):
		for j in range(N):
			pass
#############################################################


def myImageRotation(path, deg):
	img = cv.imread(path)
	img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
	assert isinstance(img, np.ndarray)

	m, n = img.shape[:2]
	xmid, ymid = int(n/2), int(m/2)
	
	out = np.zeros((m,n,3), np.float64)
	rad = deg * math.pi / 180

	inside = lambda i, j: 0 <= i < m and 0 <= j < n    

	for ni in range(m):
		for nj in range(n):
			i = ymid + (ni-ymid) * math.cos(rad) - (nj-xmid) * math.sin(rad)
			j = xmid + (nj-xmid) * math.cos(rad) + (ni-ymid) * math.sin(rad)
			r, s = int(i)-i+1, int(j)-j+1

			if inside(i,j):
				i, j = int(i), int(j) 
				out[ni][nj] = r*s*img[i][j]

				if i != m-1:
					out[ni][nj] += (1-r)*s*img[i+1][j]
				if j != n-1:
					out[ni][nj] += r*(1-s)*img[i][j+1]

				if i != m-1 and j != n-1:
					out[ni][nj] += (1-r)*(1-s)*img[i+1][j+1]

	plt.figure()
	plt.imshow(img)

	plt.figure()
	plt.imshow(out / 255)

	plt.show()	


if __name__ == '__main__':
	IMG_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

	myShrinkImageByFactorD(os.path.join(IMG_DIR, 'circles_concentric.png'), 2)
	myBilinearInterpolation(os.path.join(IMG_DIR, 'barbaraSmall.png'), 3, 2)
	myNearestNeighborInterpolation(os.path.join(IMG_DIR, 'barbaraSmall.png'), 3, 2)
	myImageRotation(os.path.join(IMG_DIR, 'barbaraSmall.png'), 30)