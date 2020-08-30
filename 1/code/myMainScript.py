import os
import math
import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1 import make_axes_locatable


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

	titles=["Original Image", "Shrunk Image"]
	
	ax = plt.subplot(1,2,1)
	im = ax.imshow(img)
	ax.set_title(titles[0])

	ax = plt.subplot(1,2,2)
	
	im = ax.imshow(out)
	ax.set_title(titles[1])
	
	divider = make_axes_locatable(ax)
	cax = divider.append_axes('right', size='5%', pad=0.05)
	
	plt.colorbar(im, cax= cax)
	plt.show()

	return out


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

	titles=["Original Image", "Bilinear Image"]
	
	ax = plt.subplot(1,2,1)
	im = ax.imshow(img)
	ax.set_title(titles[0])

	ax = plt.subplot(1,2,2)
	
	im = ax.imshow(out)
	ax.set_title(titles[1])
	
	divider = make_axes_locatable(ax)
	cax = divider.append_axes('right', size='5%', pad=0.05)
	
	plt.colorbar(im, cax= cax)
	plt.show()

	return out


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

	image_titles=["Original Image", "Neighbor Image"]
	
	ax = plt.subplot(1,2,1)
	im = ax.imshow(img)
	ax.set_title(image_titles[0])

	ax = plt.subplot(1,2,2)
	
	im = ax.imshow(out)
	ax.set_title(image_titles[1])
	
	divider = make_axes_locatable(ax)
	cax = divider.append_axes('right', size='5%', pad=0.05)
	
	plt.colorbar(im, cax= cax)
	plt.show()

	return out


def myBicubicInterpolation(path, hor, ver):
	img = cv.imread(path)
	img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
	m, n = img.shape[:2]
	M, N = hor * m - (hor - 1), ver * n - (ver - 1)
	out = np.zeros((M, N, 3), dtype=np.uint8)

	for i in range(M):
		for j in range(N):
			row, col = int(i / hor), int(j / ver)
			x, y = (i % hor) / hor, (j % ver) / ver

			if 0 < row < m - 2 and 0 < col < n - 2:
				f00, f01, f02, f03 = img[row - 1, col - 1], img[row - 1, col], img[row - 1, col + 1], img[row - 1, col + 2]
				f10, f11, f12, f13 = img[row, col - 1], img[row, col], img[row, col + 1], img[row, col + 2]
				f20, f21, f22, f23 = img[row + 1, col - 1], img[row + 1, col], img[row + 1, col + 1], img[row + 1, col + 2]
				f30, f31, f32, f33 = img[row + 2, col - 1], img[row + 2, col], img[row + 2, col + 1], img[row + 2, col + 2]

			elif row == 0 and col == 0:
				f00, f01, f02, f03 = img[row, col], img[row, col], img[row, col + 1], img[row, col + 2]
				f10, f11, f12, f13 = img[row, col], img[row, col], img[row, col + 1], img[row, col + 2]
				f20, f21, f22, f23 = img[row + 1, col], img[row + 1, col], img[row + 1, col + 1], img[row + 1, col + 2]
				f30, f31, f32, f33 = img[row + 2, col], img[row + 2, col], img[row + 2, col + 1], img[row + 2, col + 2]

			elif row == 0 and col >= m - 2:
				f00, f01, f02, f03 = img[row, col], img[row, col], img[row, col], img[row, col]
				f10, f11, f12, f13 = img[row, col], img[row, col], img[row, col], img[row, col]
				f20, f21, f22, f23 = img[row + 1, col], img[row + 1, col], img[row + 1, col], img[row + 1, col]
				f30, f31, f32, f33 = img[row + 2, col], img[row + 2, col], img[row + 2, col], img[row + 2, col]

			elif row >= n - 2 and col == 0:
				f00, f01, f02, f03 = img[row - 1, col], img[row - 1, col], img[row - 1, col + 1], img[row - 1, col + 2]
				f10, f11, f12, f13 = img[row, col], img[row, col], img[row, col + 1], img[row, col + 2]
				f20, f21, f22, f23 = img[row, col], img[row, col], img[row, col + 1], img[row, col + 2]
				f30, f31, f32, f33 = img[row, col], img[row, col], img[row, col + 1], img[row, col + 2]

			elif row >= n - 2 and col >= m - 2:
				f00, f01, f02, f03 = img[row - 1, col - 1], img[row - 1, col], img[row - 1, col], img[row - 1, col]
				f10, f11, f12, f13 = img[row, col - 1], img[row, col], img[row, col], img[row, col]
				f20, f21, f22, f23 = img[row, col - 1], img[row, col], img[row, col], img[row, col]
				f30, f31, f32, f33 = img[row, col - 1], img[row, col], img[row, col], img[row, col]

			elif row == 0:
				f00, f01, f02, f03 = img[row, col - 1], img[row, col], img[row, col + 1], img[row, col + 2]
				f10, f11, f12, f13 = img[row, col - 1], img[row, col], img[row, col + 1], img[row, col + 2]
				f20, f21, f22, f23 = img[row + 1, col - 1], img[row + 1, col], img[row + 1, col + 1], img[row + 1, col + 2]
				f30, f31, f32, f33 = img[row + 2, col - 1], img[row + 2, col], img[row + 2, col + 1], img[row + 2, col + 2]

			elif row >= n - 2:
				f00, f01, f02, f03 = img[row - 1, col - 1], img[row - 1, col], img[row - 1, col + 1], img[row - 1, col + 2]
				f10, f11, f12, f13 = img[row, col - 1], img[row, col], img[row, col + 1], img[row, col + 2]
				f20, f21, f22, f23 = img[row, col - 1], img[row, col], img[row, col + 1], img[row, col + 2]
				f30, f31, f32, f33 = img[row, col - 1], img[row, col], img[row, col + 1], img[row, col + 2]

			elif col == 0:
				f00, f01, f02, f03 = img[row - 1, col], img[row - 1, col], img[row - 1, col + 1], img[row - 1, col + 2]
				f10, f11, f12, f13 = img[row, col], img[row, col], img[row, col + 1], img[row, col + 2]
				f20, f21, f22, f23 = img[row + 1, col], img[row + 1, col], img[row + 1, col + 1], img[row + 1, col + 2]
				f30, f31, f32, f33 = img[row + 2, col], img[row + 2, col], img[row + 2, col+ 1], img[row + 2, col + 2]

			elif col >= m - 2:
				f00, f01, f02, f03 = img[row - 1, col - 1], img[row - 1, col], img[row - 1, col], img[row - 1, col]
				f10, f11, f12, f13 = img[row, col - 1], img[row, col], img[row, col], img[row, col]
				f20, f21, f22, f23 = img[row + 1, col - 1], img[row + 1, col], img[row + 1, col], img[row + 1, col]
				f30, f31, f32, f33 = img[row + 2, col - 1], img[row + 2, col], img[row + 2, col], img[row + 2, col]
			
			Z = np.array([
				f00, f01, f02, f03,
			  f10, f11, f12, f13,
			  f20, f21, f22, f23,
			  f30, f31, f32, f33
			]).reshape((4, 4, 3)).transpose()

			X = np.tile(np.array([-1, 0, 1, 2]), (4, 1))
			X[0, :] = X[0, :] ** 3
			X[1, :] = X[1, :] ** 2
			X[-1, :] = 1

			Cr = Z @ np.linalg.inv(X)
			R = np.rollaxis(Cr @ np.array([x * 3, x * 2, x, 1]), 0, 1).transpose()

			Y = np.tile(np.array([-1, 0, 1, 2]), (4, 1)).transpose()
			Y[:, 0] = Y[:, 0] ** 3
			Y[:, 1] = Y[:, 1] ** 2
			Y[:, -1] = 1

			Cc = np.linalg.inv(Y) @ R

			out[i][j] = (np.array([y * 3, y * 2, y, 1]) @ Cc)

	titles=["Original Image", "bicubic Image"]

	ax = plt.subplot(1,2,1)
	im = ax.imshow(img)
	ax.set_title(titles[0])

	ax = plt.subplot(1,2,2)

	im = ax.imshow(out)
	ax.set_title(titles[1])

	divider = make_axes_locatable(ax)
	cax = divider.append_axes('right', size='5%', pad=0.05)

	plt.colorbar(im, cax= cax)
	plt.show()

	return out


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

	out /= 255
	titles=["Original Image", "Rotated Image"]
	
	ax = plt.subplot(1,2,1)
	im = ax.imshow(img)
	ax.set_title(titles[0])

	ax = plt.subplot(1,2,2)
	
	im = ax.imshow(out)
	ax.set_title(titles[1])
	
	divider = make_axes_locatable(ax)
	cax = divider.append_axes('right', size='5%', pad=0.05)
	
	plt.colorbar(im, cax= cax)
	plt.show()

	return out


if __name__ == '__main__':
	IMG_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

	myShrinkImageByFactorD(os.path.join(IMG_DIR, 'circles_concentric.png'), 2)

	myBilinearInterpolation(os.path.join(IMG_DIR, 'barbaraSmall.png'), 3, 2)
	myNearestNeighborInterpolation(os.path.join(IMG_DIR, 'barbaraSmall.png'), 3, 2)
	myBicubicInterpolation(os.path.join(IMG_DIR, 'barbaraSmall.png'), 3, 2)
	
	myImageRotation(os.path.join(IMG_DIR, 'barbaraSmall.png'), 30)
	myImageRotation(os.path.join(IMG_DIR, 'barbaraSmall.png'), 45)
	myImageRotation(os.path.join(IMG_DIR, 'barbaraSmall.png'), 60)