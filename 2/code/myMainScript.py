import os
import math
import numpy as np 
import cv2 as cv
import matplotlib.pyplot as plt 


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


def invert(arr):
	ans = [0 for i in range(256)]
	arr = arr.astype(np.int32)
	i = 0
	for j in range(256):
		ans[i:arr[j] + 1] = [j]*(arr[j] + 1 - i)
		i = arr[j] + 1
	return ans


def myForegroundMask(path):
	if isinstance(path, str):
		img = extractImage(path)
	elif isinstance(path, np.ndarray):
		img = path
	else:
		raise TypeError('Provide image or path')

	smooth = np.zeros(img.shape, dtype=np.float64)

	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			try:
				for x in [-1,0,1]:
					for y in [-1,0,1]:
						smooth[i][j] += img[i+x][j+y]
				smooth[i][j] /= 9
			except:
				smooth[i][j] = img[i][j]

	smooth /= 255
	mask = np.array(smooth > 4/255, dtype=np.uint8)

	plt.figure()
	plt.imshow(img)

	plt.figure()
	plt.imshow(img * mask)

	plt.figure()
	plt.imshow(mask * 255) 

	plt.show()

	return img * mask


def linearStretch(x):
	assert 0 <= x <= 1

	if x < 0.3:
		return x/5
	if x > 0.7:
		return 1 - (1-x)/5
	return (x-0.3)*0.88/0.4 + 0.06


def myLinearContrastStretching(path):
	if isinstance(path, str):
		img = extractImage(path)
	elif isinstance(path, np.ndarray):
		img = path
	else:
		raise TypeError('Provide image or path')

	m, n = img.shape[:2]
	out = np.zeros(img.shape)

	for i in range(m):
		for j in range(n):
			for k in range(3):
				out[i][j][k] = linearStretch(img[i][j][k]/255)

	plt.figure()
	plt.imshow(img)
	
	plt.figure()
	plt.imshow(out) 

	plt.show()


def myHE(path):
	if isinstance(path, str):
		img = extractImage(path)
	elif isinstance(path, np.ndarray):
		img = path
	else:
		raise TypeError('Provide image or path')

	m, n = img.shape[:2]
	R = [0 for i in range(256)]
	G, B = R.copy(), R.copy()

	for i in range(m):
		for j in range(n):
			r, g, b = img[i][j]
			R[r] += 1
			G[g] += 1
			B[b] += 1

	sR, sG, sB = presums(R), presums(G), presums(B)
	fR, fG, fB = [round(255*i/(m*n)) for i in sR], [round(255*i/(m*n)) for i in sG], [round(255*i/(m*n)) for i in sB]

	out = np.zeros(img.shape, dtype=np.uint8)

	for i in range(m):
		for j in range(n):
			r, g, b = img[i][j]
			out[i][j] = np.array([fR[r], fG[g], fB[b]])

	plt.figure()
	plt.imshow(img)

	plt.figure()
	plt.imshow(out)

	plt.show()


def myHM():
	img = extractImage("../data/retina.png")
	ideal = extractImage("../data/retinaRef.png")

	px_img = np.zeros((3,256), dtype=np.int32)
	px_ideal = 	np.zeros((3,256), dtype=np.int32)

	m, n = img.shape[:2]

	for i in range(m):
		for j in range(n):
			for k in range(3):
				px_img[k][img[i][j][k]] += 1
				px_ideal[k][ideal[i][j][k]] += 1

	cdf_img = np.array([np.round(presums(row)*255/(m*n)) for row in px_img])
	cdf_ideal = np.array([np.round(presums(row)*255/(m*n)) for row in px_ideal])
	inv_ideal = np.array([invert(row) for row in cdf_ideal])

	out = np.zeros(img.shape, dtype=np.uint8)

	for i in range(m):
		for j in range(n):
			for k in range(3):
				out[i][j][k] = inv_ideal[k][int(cdf_img[k][img[i][j][k]])]

	plt.figure()
	plt.imshow(img)

	plt.figure()
	plt.imshow(ideal)

	plt.figure()
	plt.imshow(out)

	plt.show()


if __name__ == '__main__':
	IMG_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
	
	path = os.path.join(IMG_DIR, 'statue.png')
	out = myForegroundMask(path)
	
	myLinearContrastStretching(out)
	myHE(out)
	
	myHM()

	path = os.path.join(IMG_DIR, 'retina.png')
	myLinearContrastStretching(path)
	myHE(path)

	for file in os.listdir(IMG_DIR):
		if 'statue' in file or 'retina' in file:
			continue 

		print(file)
		path = os.path.join(IMG_DIR, file)
		myLinearContrastStretching(path)
		myHE(path)